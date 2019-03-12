/*Copyright (c) 2018 Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

#include <cassert>
#include <iostream>
#include <numa.h>
#include <unistd.h>


#include "join-primitives.cuh"

__global__ void init_payload (int* R, int n) {
    for (int i = threadIdx.x + blockIdx.x*blockDim.x; i < n; i += blockDim.x*gridDim.x)
        R[i] = i;
}

/*
S= keys of data to be partitioned
P= payloads of data to be partitioned
heads= keeps information on first bucket per partition and number of elements in it, packet in one 64-bit integer (only used here)
chains= the successor of a bucket in the bucket list
out_cnts= number of elements per partition
buckets_used= how many buckets are reserved by the partitioning already
offsets= describe the segments that occur due to partitioning
note: multithreaded partitioning creates partitions that consist of contiguous segments
=> iterate over these segments to avoid handling empty slots

output_S= bucketized partitions of data keys
output_P= bucketized partitions of data payloads
cnt= number of elements to partition on total
log_parts- log of number of partitions
first_bit= shift the keys before "hashing"
num_threads= number of threads used in CPU side, used together with offsets

preconditions:
heads: current bucket (1 << 18) [special value for no bucket] and -1 elements (first write allocates bucket)
out_cnts: 0
buckets_used= number of partitions (first num_parts buckets are reserved)
*/
__global__ void partition_pass_one (
                                    const int32_t   * __restrict__ S,
                                    const int32_t   * __restrict__ P,
                                    const size_t    * __restrict__ offsets,
                                          uint64_t  * __restrict__ heads,
                                          uint32_t  * __restrict__ buckets_used,
                                          uint32_t  * __restrict__ chains,
                                          uint32_t  * __restrict__ out_cnts,
                                          int32_t   * __restrict__ output_S,
                                          int32_t   * __restrict__ output_P,
                                          size_t                   cnt,
                                          uint32_t                 log_parts,
                                          uint32_t                 first_bit,
                                          uint32_t                 num_threads) {
    assert((((size_t) bucket_size) + ((size_t) blockDim.x) * gridDim.x) < (((size_t) 1) << 32));
    const uint32_t parts     = 1 << log_parts;
    const int32_t parts_mask = parts - 1;

    uint32_t * router = (uint32_t *) int_shared;

    uint32_t segment = 0;
    size_t segment_limit = offsets[1];
    size_t segment_next = offsets[2];

    size_t* shared_offsets = (size_t*) (int_shared + 1024*4 + 4*parts);

    /*if no segmentation in input use one segment with all data, else copy the segment info*/
    if (offsets != NULL) {
    	for (int i = threadIdx.x; i < 4*num_threads; i += blockDim.x) {
        	shared_offsets[i] = offsets[i];
    	}
	} else {
		for (int i = threadIdx.x; i < 4*num_threads; i += blockDim.x) {
			if (i == 1)
				shared_offsets[i] = cnt;
			else
	        	shared_offsets[i] = 0;
        }
	}

    shared_offsets[4*num_threads] = cnt+4096;
    shared_offsets[4*num_threads+1] = cnt+4096;

    /*partition element counter starts at 0*/
    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x ) 
        router[1024*4 + parts + j] = 0;
    
    if (threadIdx.x == 0) 
        router[0] = 0;

    __syncthreads();

    
    /*iterate over the segments*/
    for (int u = 0; u < 2*num_threads; u++) {
        size_t segment_start = shared_offsets[2*u];
        size_t segment_limit = shared_offsets[2*u + 1]; 
        size_t segment_end   = segment_start + ((segment_limit - segment_start + 4096 - 1)/4096)*4096;

        for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x) + segment_start; i < segment_end ; i += 4 * blockDim.x * gridDim.x) {
            vec4 thread_vals = *(reinterpret_cast<const vec4 *>(S + i));

            uint32_t thread_keys[4];

            /*compute local histogram for a chunk of 4*blockDim.x elements*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (i + k < segment_limit){
                    uint32_t partition = (hasht(thread_vals.i[k]) >> first_bit) & parts_mask;

                    atomicAdd(router + (1024 * 4 + parts + partition), 1);
                
                    thread_keys[k] = partition;
                } else {
                    thread_keys[k] = 0;
                }
            }

            __syncthreads();

            for (size_t j = threadIdx.x; j < parts ; j += blockDim.x ) {
                uint32_t cnt = router[1024 * 4 + parts + j];

                if (cnt > 0){
                    atomicAdd(out_cnts + j, cnt);
                
                    uint32_t pcnt     ;
                    uint32_t bucket   ;
                    uint32_t next_buck;

                    bool repeat = true;

                    while (__any(repeat)){
                        if (repeat){
                            /*check if any of the output bucket is filling up*/
                            uint64_t old_heads = atomicAdd(heads + j, ((uint64_t) cnt) << 32);
    
                            atomicMin(heads + j, ((uint64_t) (2*bucket_size)) << 32);

                            pcnt       = ((uint32_t) (old_heads >> 32));
                            bucket     =  (uint32_t)  old_heads        ;

                            /*now there are two cases:
                            // 2) old_heads.cnt >  bucket_size ( => locked => retry)
                            // if (pcnt       >= bucket_size) continue;*/

                            if (pcnt < bucket_size){
                                /* 1) old_heads.cnt <= bucket_size*/

                                /*check if the bucket was filled*/
                                if (pcnt + cnt >= bucket_size){
                                    if (bucket < (1 << 18)) {
                                        next_buck = atomicAdd(buckets_used, 1);                                
                                        chains[bucket]     = next_buck;
                                    } else {
                                        next_buck = j;
                                    }
                                    uint64_t tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);
    
                                    atomicExch(heads + j, tmp);
                                } else {
                                    next_buck = bucket;
                                }
    
                                repeat = false;
                            }
                        }
                    }
    
                    router[1024 * 4             + j] = atomicAdd(router, cnt);
                    router[1024 * 4 +     parts + j] = 0;//cnt;//pcnt     ;
                    router[1024 * 4 + 2 * parts + j] = (bucket    << log2_bucket_size) + pcnt;
                    router[1024 * 4 + 3 * parts + j] =  next_buck << log2_bucket_size        ;
                }
            }
    
            __syncthreads();
    
    
            uint32_t total_cnt = router[0];
    
            __syncthreads();

            /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (i + k < segment_limit)
                    thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);
            }
    
            /*write the keys in shared memory*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k) 
                if (i + k < segment_limit)
                    router[thread_keys[k]] = thread_vals.i[k];
    
            __syncthreads();
    
            int32_t thread_parts[4];

            /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (threadIdx.x + 1024 * k < total_cnt) {
                    int32_t  val       = router[threadIdx.x + 1024 * k];
                    uint32_t partition = (hasht(val) >> first_bit) & parts_mask;

                    uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                    uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];

                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                        cnt    = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                    
                    bucket += cnt;
            
                    output_S[bucket] = val;

                    thread_parts[k] = partition;
                }
            }

            __syncthreads();

            /*read payloads of original data*/
            thread_vals = *(reinterpret_cast<const vec4 *>(P + i));

            /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k) 
                if (i + k < segment_limit) {
                    router[thread_keys[k]] = thread_vals.i[k];
                }

            __syncthreads();

            /*write payloads to partition buckets in "somewhat coalesced manner"*/
            #pragma unroll
            for (int k = 0 ; k < 4 ; ++k){
                if (threadIdx.x + 1024 * k < total_cnt) {
                    int32_t  val       = router[threadIdx.x + 1024 * k];

                    int32_t partition = thread_parts[k];

                    uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                    uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];

                    if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                        uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                        cnt    = ((bucket + cnt) & bucket_size_mask);
                        bucket = next_buck;
                    }
                    bucket += cnt;
            
                    output_P[bucket] = val;
                }
            }

            if (threadIdx.x == 0) router[0] = 0;
        }
    }
}

/*
compute information for the second partitioning pass

input:
chains=points to the successor in the bucket list for each bucket (hint: we append new buckets to the end)
out_cnts=count of elements per partition
output:
chains=packed value of element count in bucket and the partition the bucket belongs to
*/
__global__ void compute_bucket_info (uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t cnt = out_cnts[p];

        while (cnt > 0) {
            uint32_t local_cnt = (cnt >= 4096)? 4096 : cnt;
            uint32_t val = (p << 13) + local_cnt;
            
            uint32_t next = chains[cur];
            chains[cur] = val;

            cur = next;
            cnt -= 4096;
        }
    }
}

/*
S= keys of data to be re-partitioned
P= payloads of data to be re-partitioned
heads= keeps information on first bucket per partition and number of elements in it, packet in one 64-bit integer (only used here)
chains= the successor of a bucket in the bucket list
out_cnts= number of elements per partition
buckets_used= how many buckets are reserved by the partitioning already
offsets= describe the segments that occur due to partitioning
note: multithreaded partitioning creates partitions that consist of contiguous segments
=> iterate over these segments to avoid handling empty slots

output_S= bucketized partitions of data keys (results)
output_P= bucketized partitions of data payloads (results)

S_log_parts- log of number of partitions for previous pass
log_parts- log of number of partitions for this pass
first_bit= shift the keys before "hashing"
bucket_num_ptr: number of input buckets

preconditions:
heads: current bucket (1 << 18) [special value for no bucket] and -1 elements (first write allocates bucket)
out_cnts: 0
buckets_used= number of partitions (first num_parts buckets are reserved)
*/
__global__ void partition_pass_two (
                                    const int32_t   * __restrict__ S,
                                    const int32_t   * __restrict__ P,
                                    const uint32_t  * __restrict__ bucket_info,
                                          uint32_t  * __restrict__ buckets_used,
                                          uint64_t  *              heads,
                                          uint32_t  * __restrict__ chains,
                                          uint32_t  * __restrict__ out_cnts,
                                          int32_t   * __restrict__ output_S,
                                          int32_t   * __restrict__ output_P,
                                          uint32_t                 S_log_parts,
                                          uint32_t                 log_parts,
                                          uint32_t                 first_bit,
                                          uint32_t  *              bucket_num_ptr) {
    assert((((size_t) bucket_size) + ((size_t) blockDim.x) * gridDim.x) < (((size_t) 1) << 32));
    const uint32_t S_parts   = 1 << S_log_parts;
    const uint32_t parts     = 1 << log_parts;
    const int32_t parts_mask = parts - 1;

    uint32_t buckets_num = *bucket_num_ptr;

    uint32_t * router = (uint32_t *) int_shared; //[1024*4 + parts];

    for (size_t j = threadIdx.x ; j < parts ; j += blockDim.x ) 
        router[1024*4 + parts + j] = 0;
    
    if (threadIdx.x == 0) 
        router[0] = 0;

    __syncthreads();

    
    /*each CUDA block processes a bucket at a time*/
    for (size_t i = blockIdx.x; i < buckets_num; i += gridDim.x) {
        uint32_t info = bucket_info[i];
        /*number of elements per bucket*/
        uint32_t cnt = info & ((1 << 13) - 1);
        /*id of original partition*/
        uint32_t pid = info >> 13;

        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(S + bucket_size * i + 4*threadIdx.x));

        uint32_t thread_keys[4];

        /*compute local histogram for the bucket*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt){
                uint32_t partition = (hasht(thread_vals.i[k]) >> first_bit) & parts_mask;

                atomicAdd(router + (1024 * 4 + parts + partition), 1);
                
                thread_keys[k] = partition;
            } else {
                thread_keys[k] = 0;
            }
        }

        __syncthreads();

        for (size_t j = threadIdx.x; j < parts ; j += blockDim.x ) {
            uint32_t cnt = router[1024 * 4 + parts + j];

            if (cnt > 0){
                atomicAdd(out_cnts + (pid << log_parts) + j, cnt);
                
                uint32_t pcnt     ;
                uint32_t bucket   ;
                uint32_t next_buck;

                bool repeat = true;

                while (__any(repeat)){
                    if (repeat){
                        uint64_t old_heads = atomicAdd(heads + (pid << log_parts) + j, ((uint64_t) cnt) << 32);
    
                        atomicMin(heads + (pid << log_parts) + j, ((uint64_t) (2*bucket_size)) << 32);

                        pcnt       = ((uint32_t) (old_heads >> 32));
                        bucket     =  (uint32_t)  old_heads        ;

                        if (pcnt < bucket_size){
                            if (pcnt + cnt >= bucket_size){
                                if (bucket < (1 << 18)) {
                                    next_buck = atomicAdd(buckets_used, 1);                                
                                    chains[bucket]     = next_buck;
                                } else {
                                    next_buck = (pid << log_parts) + j;
                                }

                                uint64_t tmp =  next_buck + (((uint64_t) (pcnt + cnt - bucket_size)) << 32);

                                atomicExch(heads + (pid << log_parts) + j, tmp);
                            } else {
                                next_buck = bucket;
                            }
    
                            repeat = false;
                        }
                    }
                }
    
                router[1024 * 4             + j] = atomicAdd(router, cnt);
                router[1024 * 4 +     parts + j] = 0;
                router[1024 * 4 + 2 * parts + j] = (bucket    << log2_bucket_size) + pcnt;
                router[1024 * 4 + 3 * parts + j] =  next_buck << log2_bucket_size        ;
            }
        }

        __syncthreads();
    
    
        uint32_t total_cnt = router[0];
    
        __syncthreads();

        /*calculate write positions for block-wise shuffle => atomicAdd on start of partition*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (4*threadIdx.x + k < cnt)
                thread_keys[k] = atomicAdd(router + (1024 * 4 + thread_keys[k]), 1);
        }
    
        /*write the keys in shared memory*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) 
            if (4*threadIdx.x + k < cnt)
                router[thread_keys[k]] = thread_vals.i[k];
    
        __syncthreads();
    
        int32_t thread_parts[4];

        /*read shuffled keys and write them to output partitions "somewhat" coalesced*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t  val       = router[threadIdx.x + 1024 * k];
                uint32_t partition = (hasht(val) >> first_bit) & parts_mask;

                uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                    
                bucket += cnt;
            
                output_S[bucket] = val;

                thread_parts[k] = partition;
            }
        }

        __syncthreads();

        /*read payloads of original data*/
        thread_vals = *(reinterpret_cast<const vec4 *>(P + i*bucket_size + 4*threadIdx.x));

        /*shuffle payloads in shared memory, in the same offsets that we used for their corresponding keys*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k) 
            if (4*threadIdx.x + k < cnt) {
                router[thread_keys[k]] = thread_vals.i[k];
            }

        __syncthreads();

        /*write payloads to partition buckets in "somewhat coalesced manner"*/
        #pragma unroll
        for (int k = 0 ; k < 4 ; ++k){
            if (threadIdx.x + 1024 * k < total_cnt) {
                int32_t  val       = router[threadIdx.x + 1024 * k];

                int32_t partition = thread_parts[k];

                uint32_t cnt       = router[1024 * 4 +             partition] - (threadIdx.x + 1024 * k);

                uint32_t bucket    = router[1024 * 4 + 2 * parts + partition];

                if (((bucket + cnt) ^ bucket) & ~bucket_size_mask){
                    uint32_t next_buck = router[1024 * 4 + 3 * parts + partition];
                    cnt    = ((bucket + cnt) & bucket_size_mask);
                    bucket = next_buck;
                }
                bucket += cnt;
           
                output_P[bucket] = val;
            }
        }

        if (threadIdx.x == 0) router[0] = 0;
    }
}

#define LOCAL_BUCKETS_BITS 10
#define LOCAL_BUCKETS ((1 << LOCAL_BUCKETS_BITS))

#define MAX_BIT 32

__device__ int ctzd (int x) {
    if (x == 0)
        return 32;
    
    int n = 0;

    if ((n & 0x0000FFFF) == 0) {
        n += 16;
        x >>= 16;
    }

    if ((n & 0x000000FF) == 0) {
        n += 8;
        x >>= 8;
    }

    if ((n & 0x0000000F) == 0) {
        n += 4;
        x >>= 4;
    }

    if ((n & 0x00000003) == 0) {
        n += 2;
        x >>= 2;
    }

    if ((n & 0x00000001) == 0) {
        n += 1;
        x >>= 1;
    }

    return n;
}


__global__ void init_metadata_double ( 
                                uint64_t  * __restrict__ heads1,
                                uint32_t  * __restrict__ buckets_used1,
                                uint32_t  * __restrict__ chains1,
                                uint32_t  * __restrict__ out_cnts1,
                                uint32_t parts1,
                                uint32_t buckets_num1,
                                uint64_t  * __restrict__ heads2,
                                uint32_t  * __restrict__ buckets_used2,
                                uint32_t  * __restrict__ chains2,
                                uint32_t  * __restrict__ out_cnts2,
                                uint32_t parts2,
                                uint32_t buckets_num2
                                ) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    for (int i = tid; i < buckets_num1; i += blockDim.x*gridDim.x)
        chains1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        out_cnts1[i] = 0;

    for (int i = tid; i < parts1; i += blockDim.x*gridDim.x)
        heads1[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used1 = parts1;
    }

    for (int i = tid; i < buckets_num2; i += blockDim.x*gridDim.x)
        chains2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        out_cnts2[i] = 0;

    for (int i = tid; i < parts2; i += blockDim.x*gridDim.x)
        heads2[i] = (1 << 18) + (((uint64_t) bucket_size_mask) << 32);

    if (tid == 0) {
        *buckets_used2 = parts2;
    }
}

/*
Building phase for non-partitioned hash join with perfect hashing (so this property is reflected in the code, we don't follow chains), it is the best case for non-partitioned

data=array of the keys
payload=array of payloads
n=number of tuples
lookup=lookup table/hashtable that we build => we store the payload at position lookup[key]
*/
__global__ void build_perfect_array (int32_t* data, int32_t* payload, int n, int32_t* lookup) {
    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));
        vec4 thread_payloads = *(reinterpret_cast<const vec4 *>(payload + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int32_t val = thread_vals.i[k];
            int32_t payload = thread_payloads.i[k];
            lookup[val] = payload + 1;      
        }
    }
}

/*Probing phase for non-partitioned hash join with perfect hashing

data=keys for probe side
payload=payloads for probe side
n=number of elements
lookup=hashtable
aggr=the memory location in which we aggregate with atomics at the end*/
__global__ void probe_perfect_array (int32_t* data, int32_t* payload, int n, int32_t* lookup, int* aggr) {
    int count = 0;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));
        vec4 thread_payloads = *(reinterpret_cast<const vec4 *>(payload + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];
            int payload = thread_payloads.i[k];
            int res = lookup[val];

            if (res)
                count += (payload * (res - 1));    
        }
    }

    atomicAdd(aggr, count);
}


/*
Building phase for non-partitioned hash join with chaining

data=array of the keys
payload=array of payloads
n=number of tuples
log_parts=log size of hashtable/chains
output=the chains [the rest of the array stays in place]
head=the first element of each chain
*/
__global__ void build_ht_chains (int32_t* data, int n, uint32_t log_parts, int32_t* output, int* head) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];
            int hval = val & parts_mask;

            int last = atomicExch(head + hval, i+k+1);
            //int64_t wr = (((int64_t) last) << 32) + val;
            output[i + k] = last;         
        }
    }
}

/*
Probing phase for non-partitioned hash join with chaining

data=array of the keys
payload=array of payloads
n=number of tuples
log_parts=log size of hashtable/chains
ht=the chains that show the successor for each build element
head=the first element of each chain
ht_key=the keys of the hashtable as an array
ht_pay=the payloads of the hashtable as an array
aggr=the memory location in which we aggregate with atomics at the end
*/
__global__ void chains_probing (int32_t* data, int32_t* payload, int n, uint32_t log_parts, int32_t* ht, int32_t* ht_key, int32_t* ht_pay, int* head, int* aggr) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;
    int count = 0;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));
        vec4 thread_payloads = *(reinterpret_cast<const vec4 *>(payload + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];
            int payload = thread_payloads.i[k];
            int hval = val & parts_mask;

            int next = head[hval];

            while (next != 0) {
                int ht_val = ht_key[next-1];

                if (ht_val == val)
                    count += (payload * ht_pay[next-1]);

                next = ht[next-1];
            }       
        }
    }

    atomicAdd(aggr, count);
}


/*functions for linear probing

FIXME: there is a bug so it is not operational yet [was not in paper so this is not urgent]
*/

__global__ void ht_hist (int* data, int n, int log_parts, int* hist) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];
            int hval = val & parts_mask;

            int off = atomicAdd(hist + hval, 1);
        }
    }
}

__global__ void ht_offsets (int log_parts, int* hist, int* offset, int* aggr) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < parts; i += blockDim.x * gridDim.x) {
        int cur = hist[i];
        int off = atomicAdd(aggr, cur);
        hist[i] = off;
        offset[i] = off;
    }
} 

__global__ void build_ht_linear (int* data, int* payload, size_t n, int log_parts, int* offset, int* ht, int* htp) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));
        vec4 thread_payloads = *(reinterpret_cast<const vec4 *>(payload + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];
            int hval = val & parts_mask;

            int off = atomicAdd(offset + hval, 1);

            ht[off] = val;
            htp[off] = thread_payloads.i[k];

        }
    }
}

__global__ void linear_probing (int* data, int* payload, int* ht, int* htp, int* offset_s, int* offset_e, size_t n, int log_parts, int* aggr) {
    int parts = 1 << log_parts;
    int parts_mask = parts-1;
    int count = 0;

    for (size_t i = 4 *(threadIdx.x + blockIdx.x * blockDim.x); i < n ; i += 4 * blockDim.x * gridDim.x){
        vec4 thread_vals = *(reinterpret_cast<const vec4 *>(data + i));
        vec4 thread_payloads = *(reinterpret_cast<const vec4 *>(payload + i));

        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            int val = thread_vals.i[k];

            for (int j = 0; j < 32; j++) {
                int probe = __shfl(val, j);
                int pay = __shfl(thread_payloads.i[k], j);
                int hval = probe & parts_mask;

                int start = offset_s[hval];
                int end = offset_e[hval];

                for (int p = start + threadIdx.x % 32; p < end; p += 32) {
                    if (ht[p] == probe) {
                        count += pay*htp[p];
                    }
                }
            }
        }
    }

    atomicAdd(aggr, count);
}

/*break "long" bucket chains to smaller chains
this helps load balancing because we can allocate work at sub-chain granularity
and effectively solve the skew problem

bucket_info=we store the packed (partition, element count) value for each bucket
chains=successor in partition's bucket list
out_cnts=count of elements in this partition
log_parts= log of number of partitions
threshold=the maximum number of elements per subchain*/
__global__ void decompose_chains (uint32_t* bucket_info, uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts, int threshold) {
    uint32_t parts = 1 << log_parts;

    for (int p = threadIdx.x + blockIdx.x*blockDim.x; p < parts; p += gridDim.x*blockDim.x) {
        uint32_t cur = p;
        int32_t  cnt = out_cnts[p];
        uint32_t first_cnt = (cnt >= threshold)? threshold : cnt;
        int32_t  cutoff = 0; 

        while (cnt > 0) {
            cutoff += bucket_size;
            cnt -= bucket_size;

            uint32_t next = chains[cur];
            
            if (cutoff >= threshold && cnt > 0) {
                uint32_t local_cnt = (cnt >= threshold)? threshold : cnt;

                bucket_info[next] = (p << 15) + local_cnt;
                chains[cur] = 0;
                cutoff = 0;
            } else if (next != 0) {
                bucket_info[next] = 0;
            }


            cur = next;
        }

        bucket_info[p] = (p << 15) + first_cnt;
    }
}

/*kernel for performing the join between the partitioned relations

R,Pr= bucketized keys and payloads for relation R (probe side)
S,Ps= buckerized keys and payloads for relation S (build side)
bucket_info=the info that tells us which partition each bucket belongs to, the number of elements (or whether it belongs to a chain)
S_cnts, S_chain= for build-side we don't pack the info since we operate under the assumption that it is usually one bucket per partition (we don't load balance)
buckets_num=number of buckets for R
results=the memory address where we aggregate
*/
__global__ void join_partitioned_aggregate (
                                    const int32_t*               R,
                                    const int32_t*               Pr,
                                    const uint32_t*              R_chain,
                                    const uint32_t*              bucket_info,
                                    const int32_t*               S,
                                    const int32_t*               Ps,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    uint32_t*                    buckets_num,
                                    int32_t*                     results) {

    /*in order to saze space, we discard the partitioning bits, then we can try fitting keys in int16_t [HACK]*/
    __shared__ int16_t elem[4096 + 512];
    __shared__ int32_t payload[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];


    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    int lid = tid % 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int buckets_cnt = *buckets_num;

    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0) {
            /*unpack information on the subchain*/
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);

            int len_S = S_cnts[p];

            /*S partition doesn't fit in shared memory*/
            if (len_S > 4096+512) {
                int bucket_r_loop = bucket_r;

                /*now we will build a bucket of R side in the shared memory at a time and then probe it with S-side
                sensible because
                1) we have guarantees on size of R from the chain decomposition
                2) this is a skewed scenario so size of S can be arbitrary*/
                for (int offset_r = 0; offset_r < len_R; offset_r += bucket_size) {
                    for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                        head[i] = -1;
                    __syncthreads();

                    /*build a hashtable from an R bucket*/
                    for (int base_r = 0; base_r < bucket_size; base_r += 4*blockDim.x) {
                        vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                        vec4 data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                        int l_cnt_R = len_R - offset_r - base_r - 4 * threadIdx.x;

                        int cnt = 0;                    

                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            if (k < l_cnt_R) {
                                int val = data_R.i[k];
                                elem[base_r + k*blockDim.x + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                                payload[base_r + k*blockDim.x + tid] = data_Pr.i[k];
                                int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                                int32_t last = atomicExch(&head[hval], base_r + k*blockDim.x + tid);
                                next[base_r + k*blockDim.x + tid] = last;
                            }
                        }
                    }

                    bucket_r_loop = R_chain[bucket_r_loop];

                    __syncthreads();

                    int bucket_s_loop = p;
                    int base_s = 0;
        
                    /*probe hashtable from an S bucket*/
                    for (int offset_s = 0; offset_s < len_S; offset_s += 4*blockDim.x) {
                        vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                        vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                        int l_cnt_S = len_S - offset_s - 4 * threadIdx.x;

                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            int32_t val = data_S.i[k];
                            int32_t pval = data_Ps.i[k];
                            int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                            int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);

                            if (k < l_cnt_S) {
                                int32_t pos = head[hval];
                                while (pos >= 0) {
                                    if (elem[pos] == tval) {
                                        count += pval*payload[pos];
                                    }

                                    pos = next[pos];
                                }
                            }                   
                        }

                        base_s += 4*blockDim.x;
                        if (base_s >= bucket_size) {
                            bucket_s_loop = S_chain[bucket_s_loop];
                            base_s = 0;
                        }
                    }

                    __syncthreads();
                }
            } else {
                for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                    head[i] = -1;

                int rem_s = len_S % 4096;
                rem_s = (rem_s + 4 - 1)/4;

                __syncthreads();

                int off;
                int it;
                int base = 0;

                it = p;
                off = 0;

                /*build hashtable for S-side*/
                for (off = 0; off < len_S;) {
                    vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * it + base + 4*threadIdx.x));
                    vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * it + base +4*threadIdx.x));
                    int l_cnt_S = len_S - off - 4 * threadIdx.x;

                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if (k < l_cnt_S) {
                            int val = data_S.i[k];
                            elem[off + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                            payload[off + tid] = data_Ps.i[k];
                            int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                            int32_t last = atomicExch(&head[hval], off + tid);
                            next[off + tid] = last;
                        }   

                        off += (off < bucket_size)? blockDim.x : rem_s;
                    }

                    if (base >= bucket_size) {
                        it = S_chain[it];  
                        base = 0;
                    }


                }

                __syncthreads();

                it = bucket_r;
                off = 0;

                /*probe from R-side*/
                for (; 0 < len_R; off += 4*blockDim.x, len_R -= 4*blockDim.x) {
                    vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * it + off + 4*threadIdx.x));
                    vec4 data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * it + off + 4*threadIdx.x));
                    int l_cnt_R = len_R - 4 * threadIdx.x;

                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int32_t val = data_R.i[k];
                        int32_t pval = data_Pr.i[k];
                        /*hack to fit more data in shared memory*/
                        int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);

                        if (k < l_cnt_R) {
                            int32_t pos = head[hval];
                            while (pos >= 0) {
                                if (elem[pos] == tval) {
                                    count += pval*payload[pos];
                                }

                                pos = next[pos];
                            }
                        }                   
                    }

                    if (off >= bucket_size) {
                        it = R_chain[it];
                        off = 0;
                    }
                }

                __syncthreads();
            }
        }
    }

    atomicAdd(results, count);

    __syncthreads();
}

/*maximum size of output, we always write at *write_offset MOD (FOLD+1)*
we use it in order to simulate the cases that output size explodes. we do the actual writes then overwrite them*/
#define FOLD ((1 << 24) - 1)
/*the number of elements that can be stored in a warp-level buffer during the join materialization*/
#define SHUFFLE_SIZE 16


/*practically the same as join_partitioned_aggregate

i add extra comments for the materialization technique*/
__global__ void join_partitioned_results (
                                    const int32_t*               R,
                                    const int32_t*               Pr,
                                    const uint32_t*              R_chain,
                                    const uint32_t*              bucket_info,
                                    const int32_t*               S,
                                    const int32_t*               Ps,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    uint32_t*                    buckets_num,
                                    int32_t*                     results,
                                    int32_t*                     output) {
    __shared__ int16_t elem[4096 + 512];
    __shared__ int32_t payload[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];
    __shared__ int32_t shuffle[2*SHUFFLE_SIZE*32];


    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    int lid = tid % 32;
    int gid = tid / 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int ptr;

    int threadmask = (lid < 31)? ~((1 << (lid+1)) - 1) : 0;

    int shuffle_ptr = 0;

    int32_t* warp_shuffle = shuffle + gid * 2 * SHUFFLE_SIZE;

    int buckets_cnt = *buckets_num;


    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0) { 
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);
            int len_S = S_cnts[p];

            if (len_S > 4096+512) {
                int bucket_r_loop = bucket_r;

                for (int offset_r = 0; offset_r < len_R; offset_r += bucket_size) {
                    for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                        head[i] = -1;
                    __syncthreads();

                    for (int base_r = 0; base_r < bucket_size; base_r += 4*blockDim.x) {
                        vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                        vec4 data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * bucket_r_loop + base_r + 4*threadIdx.x));
                        int l_cnt_R = len_R - offset_r - base_r - 4 * threadIdx.x;

                        int cnt = 0;                    

                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            if (k < l_cnt_R) {
                                int val = data_R.i[k];
                                elem[base_r + k*blockDim.x + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                                payload[base_r + k*blockDim.x + tid] = data_Pr.i[k];
                                int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                                int32_t last = atomicExch(&head[hval], base_r + k*blockDim.x + tid);
                                next[base_r + k*blockDim.x + tid] = last;
                            }
                        }
                    }

                    bucket_r_loop = R_chain[bucket_r_loop];

                    __syncthreads();

                    int bucket_s_loop = p;
                    int base_s = 0;

                    for (int offset_s = 0; offset_s < len_S; offset_s += 4*blockDim.x) {
                        vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                        vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * bucket_s_loop + base_s + 4*threadIdx.x));
                        int l_cnt_S = len_S - offset_s - 4 * threadIdx.x;

                        #pragma unroll
                        for (int k = 0; k < 4; k++) {
                            int32_t val = data_S.i[k];
                            int32_t pval = data_Ps.i[k];
                            int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                            int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);
                            int32_t pay;

                            int32_t pos = (k < l_cnt_S)? head[hval] : -1;

                            /*check at warp level whether someone is still following chain => this way we can shuffle without risk*/
                            int pred = (pos >= 0);

                            while (__any(pred)) {
                                int wr_intention = 0;

                                /*we have a match, fetch the data to be written*/
                                if (pred) {
                                    if (elem[pos] == tval) {
                                        pay = payload[pos];
                                        wr_intention = 1;
                                        count++;
                                    }

                                    pos = next[pos];
                                    pred = (pos >= 0);
                                }

                                /*find out who had a match in this execution step*/
                                int mask = __ballot(wr_intention);

                                /*our software managed buffer will overflow, flush it*/
                                int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
                                shuffle_ptr = shuffle_ptr + __popc(mask);
                                
                                /*while it overflows, flush
                                we flush 16 keys and then the 16 corresponding payloads consecutively, of course other formats might be friendlier*/
                                while (shuffle_ptr >= SHUFFLE_SIZE) {
                                    if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                                        warp_shuffle[wr_offset] = pay;
                                        warp_shuffle[wr_offset+SHUFFLE_SIZE] = pval;
                                        wr_intention = 0;
                                    }

                                   if (lid == 0) {
                                        ptr = atomicAdd(results, 2*SHUFFLE_SIZE);
                                        ptr = ptr & FOLD;
                                   }

                                    ptr = __shfl(ptr, 0);

                                    output[ptr + lid] = warp_shuffle[lid];

                                    wr_offset -= SHUFFLE_SIZE;
                                    shuffle_ptr -= SHUFFLE_SIZE;
                                }

                                /*now the fit, write them in buffer*/
                                if (wr_intention && (wr_offset >= 0)) {
                                    warp_shuffle[wr_offset] = pay;
                                    warp_shuffle[wr_offset+SHUFFLE_SIZE] = pval;
                                    wr_intention = 0;
                                }
                            }                   
                        }

                        base_s += 4*blockDim.x;
                        if (base_s >= bucket_size) {
                            bucket_s_loop = S_chain[bucket_s_loop];
                            base_s = 0;
                        }
                    }

                    __syncthreads();
                }
            } else {
                for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                    head[i] = -1;

                int rem_s = len_S % 4096;
                rem_s = (rem_s + 4 - 1)/4;

                __syncthreads();

                int off;
                int it;
                int base = 0;

                it = p;
                off = 0;


                for (off = 0; off < len_S;) {
                    vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * it + base + 4*threadIdx.x));
                    vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * it + base +4*threadIdx.x));
                    int l_cnt_S = len_S - off - 4 * threadIdx.x;

                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        if (k < l_cnt_S) {
                            int val = data_S.i[k];
                            elem[off + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                            payload[off + tid] = data_Ps.i[k];
                            int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                            int32_t last = atomicExch(&head[hval], off + tid);
                            next[off + tid] = last;
                        }   

                        off += (off < bucket_size)? blockDim.x : rem_s;
                        base += blockDim.x;
                    }

                    if (base >= bucket_size) {
                        it = S_chain[it];  
                        base = 0;
                    }
                }

                __syncthreads();

                it = bucket_r;
                off = 0;

                for (; 0 < len_R; off += 4*blockDim.x, len_R -= 4*blockDim.x) {
                    int l_cnt_R = len_R - 4 * threadIdx.x;
                    vec4 data_R;
                    vec4 data_Pr;

                    data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * it + off + 4*threadIdx.x));
                    data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * it + off + 4*threadIdx.x));

                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        int32_t val = data_R.i[k];
                        int32_t pval = data_Pr.i[k];
                        int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);
                        int32_t pay;

                        int32_t pos = (k < l_cnt_R)? head[hval] : -1;

                        /*same as previous code block*/
                        int pred = (pos >= 0);

                        while (__any(pred)) {
                            int wr_intention = 0;

                            if (pred) {
                                if (elem[pos] == tval) {
                                    pay = payload[pos];
                                    wr_intention = 1;
                                    count++;
                                }

                                pos = next[pos];
                                pred = (pos >= 0);
                            }

                            int mask = __ballot(wr_intention);

                            int wr_offset = shuffle_ptr +  __popc(mask & threadmask);
                            shuffle_ptr = shuffle_ptr + __popc(mask);
                                
                            while (shuffle_ptr >= SHUFFLE_SIZE) {
                                if (wr_intention && (wr_offset < SHUFFLE_SIZE)) {
                                    warp_shuffle[wr_offset] = pval;
                                    warp_shuffle[wr_offset+SHUFFLE_SIZE] = pay;
                                    wr_intention = 0;
                                }

                                if (lid == 0) {
                                    ptr = atomicAdd(results, 2*SHUFFLE_SIZE);

                                    ptr = ptr & FOLD;
                                }

                                ptr = __shfl(ptr, 0);

                                output[ptr + lid] = warp_shuffle[lid];

                                wr_offset -= SHUFFLE_SIZE;
                                shuffle_ptr -= SHUFFLE_SIZE;
                            }

                            if (wr_intention && (wr_offset >= 0)) {
                                warp_shuffle[wr_offset] = pval;
                                warp_shuffle[wr_offset+SHUFFLE_SIZE] = pay;
                                wr_intention = 0;
                            }
                        }                   
                    }

                    if (off >= bucket_size) {
                        it = R_chain[it];
                        off = 0;
                    }
                }

                __syncthreads();
            }
        }
    }

    if (lid == 0) {
        ptr = atomicAdd(results, 2*shuffle_ptr);
        ptr = ptr & FOLD;
    }

    ptr = __shfl(ptr, 0);

    if (lid < shuffle_ptr) {
        output[ptr + lid] = warp_shuffle[lid];
        output[ptr + lid + shuffle_ptr] = warp_shuffle[lid + SHUFFLE_SIZE]; 
    }

    __syncthreads();
}

/*again the same but payload is the virtual tuple id and we late materialize from Dx arrays which store the actual columns that we need
also here we have no overflows because if we did, we wouldn't fit the data/extra columns :) */
__global__ void join_partitioned_varpayload (
                                    const int32_t*               R,
                                    const int32_t*               Pr,
                                    const int32_t*               Dr,
                                    const uint32_t*              R_chain,
                                    const uint32_t*              bucket_info,
                                    const int32_t*               S,
                                    const int32_t*               Ps,
                                    const int32_t*               Ds,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    int32_t                      col_num1,
                                    int32_t                      col_num2,
                                    int32_t                      rel_size,
                                    uint32_t*                    buckets_num,
                                    int32_t*                     results) {
    __shared__ int16_t elem[4096 + 512];
    __shared__ int32_t payload[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];


    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    int lid = tid % 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int buckets_cnt = *buckets_num;

    for (uint32_t bucket_r = block; bucket_r < buckets_cnt; bucket_r += pwidth) {
        int info = bucket_info[bucket_r];

        if (info != 0)  {
            int p = info >> 15;
            int len_R = info & ((1 << 15) - 1);

            int len_S = S_cnts[p];

            for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                head[i] = -1;

            int rem_s = len_S % 4096;
            rem_s = (rem_s + 4 - 1)/4;

            __syncthreads();

            int off;
            int it;
            int base = 0;

            it = p;
            off = 0;

            for (off = 0; off < len_S;) {
                vec4 data_S = *(reinterpret_cast<const vec4 *>(S + bucket_size * it + base + 4*threadIdx.x));
                vec4 data_Ps = *(reinterpret_cast<const vec4 *>(Ps + bucket_size * it + base +4*threadIdx.x));
                int l_cnt_S = len_S - off - 4 * threadIdx.x;

                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    if (k < l_cnt_S) {
                        int val = data_S.i[k];
                        elem[off + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        payload[off + tid] = data_Ps.i[k];
                        int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                        int32_t last = atomicExch(&head[hval], off + tid);
                        next[off + tid] = last;
                    }   

                    off += (off < bucket_size)? blockDim.x : rem_s;
                }

                if (base >= bucket_size) {
                    it = S_chain[it];  
                    base = 0;
                }


            }

            __syncthreads();

            it = bucket_r;
            off = 0;

            for (; 0 < len_R; off += 4*blockDim.x, len_R -= 4*blockDim.x) {
                vec4 data_R = *(reinterpret_cast<const vec4 *>(R + bucket_size * it + off + 4*threadIdx.x));
                vec4 data_Pr = *(reinterpret_cast<const vec4 *>(Pr + bucket_size * it + off + 4*threadIdx.x));
                int l_cnt_R = len_R - 4 * threadIdx.x;

                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int32_t val = data_R.i[k];
                    int32_t pval = data_Pr.i[k];
                    int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                    int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);

                    if (k < l_cnt_R) {
                        int32_t pos = head[hval];
                        while (pos >= 0) {
                            if (elem[pos] == tval) {
                                int32_t bval = payload[pos];

                                for (int z = 0; z < col_num1; z++)
                                    count += Dr[pval + z*rel_size];

                                for (int z = 0; z < col_num2; z++)
                                    count += Ds[bval + z*rel_size];
                            }

                            pos = next[pos];
                        }
                    }                   
                }

                if (off >= bucket_size) {
                    it = R_chain[it];
                    off = 0;
                }
            }

            __syncthreads();

        }
    }

    atomicAdd(results, count);

    __syncthreads();
}

/*late materialization and perfect hashing*/
__global__ void probe_perfect_array_varpay (int32_t* data, int32_t* Dr, int n, int32_t* lookup, int32_t* Ds, int col_num1, int col_num2, int rel_size, int* aggr) {
    int count = 0;

    for (size_t i = threadIdx.x + blockIdx.x * blockDim.x; i < n ; i += blockDim.x * gridDim.x) {
        int val = data[i];
        int payload = i;
        int res = lookup[val];

        if (res > 0) {
            res--;

            for (int z = 0; z < col_num1; z++)
                count += Dr[payload + z*rel_size];
            for (int z = 0; z < col_num2; z++)
                count += Ds[res + z*rel_size];
        }
    }

    atomicAdd(aggr, count);
}

/*partition and compute metadata for relation with key+payload*/
void prepare_Relation_payload (int* R, int* R_temp, int* P, int* P_temp, size_t RelsNum, uint32_t buckets_num, uint64_t* heads[2], uint32_t* cnts[2], uint32_t* chains[2], uint32_t* buckets_used[2], uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit, cudaStream_t streams, size_t* offsets_GPU, uint32_t num_threads) {
    init_metadata_double<<<64, 1024, 0, streams>>> (
        heads[0], buckets_used[0], chains[0], cnts[0], 1 << log_parts1, buckets_num,
        heads[1], buckets_used[1], chains[1], cnts[1], 1 << (log_parts1 + log_parts2), buckets_num
    );

    partition_pass_one <<<64, 1024, (1024*4 + 4*(1 << log_parts1)) * sizeof(int32_t) + (4*num_threads+2)*sizeof(size_t), streams>>>(
                                                R, P,
                                                offsets_GPU,
                                                heads[0],
                                                buckets_used[0],
                                                chains[0],
                                                cnts[0],
                                                R_temp, P_temp,
                                                RelsNum,
                                                log_parts1,
                                                first_bit + log_parts2,
                                                num_threads
    );


    compute_bucket_info <<<64, 1024, 0, streams>>> (chains[0], cnts[0], log_parts1);

    partition_pass_two <<<64, 1024, (1024*4 + 4*(1 << log_parts2)) * sizeof(int32_t) + ((2 * (1 << log_parts2) + 1)* sizeof(int32_t)), streams>>>(
                                    R_temp, P_temp,
                                    chains[0],
                                    buckets_used[1], heads[1], chains[1], cnts[1],
                                    R, P,
                                    log_parts1, log_parts2, first_bit,
                                    buckets_used[0]);

}

/*partition and compute metadata for relation with key+payload. We use different buffers at the end (it makes sense for UVA based techniques)*/
void prepare_Relation_payload_triple (int* R, int* R_temp, int* R_final, int* P, int* P_temp, int* P_final, size_t RelsNum, uint32_t buckets_num, uint64_t* heads[2], uint32_t* cnts[2], uint32_t* chains[2], uint32_t* buckets_used[2], uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit, cudaStream_t streams, size_t* offsets_GPU, uint32_t num_threads) {
    init_metadata_double<<<64, 1024, 0, streams>>> (
        heads[0], buckets_used[0], chains[0], cnts[0], 1 << log_parts1, buckets_num,
        heads[1], buckets_used[1], chains[1], cnts[1], 1 << (log_parts1 + log_parts2), buckets_num
    );

    partition_pass_one <<<64, 1024, (1024*4 + 4*(1 << log_parts1)) * sizeof(int32_t) + (4*num_threads+2)*sizeof(size_t), streams>>>(
                                                R, P,
                                                offsets_GPU,
                                                heads[0],
                                                buckets_used[0],
                                                chains[0],
                                                cnts[0],
                                                R_temp, P_temp,
                                                RelsNum,
                                                log_parts1,
                                                first_bit + log_parts2,
                                                num_threads
    );

    CHK_ERROR(cudaDeviceSynchronize());


    compute_bucket_info <<<64, 1024, 0, streams>>> (chains[0], cnts[0], log_parts1);

    partition_pass_two <<<64, 1024, (1024*4 + 4*(1 << log_parts2)) * sizeof(int32_t) + ((2 * (1 << log_parts2) + 1)* sizeof(int32_t)), streams>>>(
                                    R_temp, P_temp,
                                    chains[0],
                                    buckets_used[1], heads[1], chains[1], cnts[1],
                                    R_final, P_final,
                                    log_parts1, log_parts2, first_bit,
                                    buckets_used[0]);

    

}

template <typename Tv>
struct chain_iterator_ref_generic{
    Tv   x  ;
    int  cnt;
};

template <typename T, typename Tv>
class chain_iterator_generic{
private:
    const T         * __restrict__ S_parts      ;
    const uint32_t  * __restrict__ S_chains     ;
    const uint32_t cnt ;
    
    const T         * __restrict__ ptr          ;
    
    uint32_t current_bucket                     ;
    uint32_t next_bucket                        ;
    uint32_t i                                  ;
public:
    __device__ __forceinline__ chain_iterator_generic(
                    const T         * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 current_partition):
        S_parts(S_parts + (16/sizeof(T)) * threadIdx.x), S_chains(S_chains), 
        cnt((S_cnts[current_partition]/((16/sizeof(T)) * blockDim.x))*(16/sizeof(T)) + max(((int32_t) (S_cnts[current_partition] % ((16/sizeof(T)) * blockDim.x))) - ((int32_t) ((16/sizeof(T)) * threadIdx.x)), 0)), 
        ptr(S_parts + ((size_t) current_partition << log2_bucket_size) + (16/sizeof(T)) * threadIdx.x), 
        current_bucket(current_partition), 
        next_bucket(S_chains[current_partition]), 
        i(0){}

    __device__ __forceinline__ chain_iterator_generic(
                    const uint32_t  * __restrict__ S_cnts,
                          uint32_t                 current_partition):
       cnt(0), 
       i(((S_cnts[current_partition] + (16/sizeof(T)) * blockDim.x - 1)/((16/sizeof(T)) * blockDim.x))*(16/sizeof(T))){}

    __device__ __forceinline__ chain_iterator_generic<T, Tv>& operator++(){
        i   += (16/sizeof(T));// * blockDim.x;
        ptr += (16/sizeof(T)) * blockDim.x;

        if ((i * blockDim.x) & bucket_size_mask) return *this;
        
        current_bucket = next_bucket;//int_shared[0];

        ptr = S_parts + (current_bucket << log2_bucket_size);

        next_bucket = S_chains[next_bucket];

        return *this;
    }

    __device__ __forceinline__ chain_iterator_ref_generic<Tv> operator*() const {
        chain_iterator_ref_generic<Tv> tmp;
        tmp.x   = *reinterpret_cast<const Tv *>(ptr);
        tmp.cnt = cnt - i;
        return tmp;
    }

    __device__ __forceinline__ bool operator!=(const chain_iterator_generic<T, Tv>& o){
        return i != o.i;
    }
};

template <typename T, typename Tv>
class chain_generic{
private:
    const T         * __restrict__ S_parts  ;
    const uint32_t  * __restrict__ S_cnts   ;
    const uint32_t  * __restrict__ S_chains ;
    const uint32_t                 partition;
public:
    __device__ __host__ __forceinline__ chain_generic(
                    const T         * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 partition):
        S_parts(S_parts), S_cnts(S_cnts), S_chains(S_chains), partition(partition){}

    __device__ __forceinline__ chain_iterator_generic<T, Tv> begin() const {
        return chain_iterator_generic<T, Tv>(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __forceinline__ chain_iterator_generic<T, Tv> end() const {
        return chain_iterator_generic<T, Tv>(S_cnts, partition);
    }
};

template <typename T, typename Tv>
class chains_generic {
private:
    const T         * __restrict__ S_parts  ;
    const uint32_t  * __restrict__ S_cnts   ;
    const uint32_t  * __restrict__ S_chains ;
public:
    __device__ __host__ __forceinline__ chains_generic(
                    const T         * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ):
        S_parts(S_parts), S_cnts(S_cnts), S_chains(S_chains){}

    __device__ __host__ __forceinline__ chain_generic<T, Tv> get_chain(uint32_t partition) const{
        return chain_generic<T, Tv>(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __forceinline__ uint32_t get_chain_size(uint32_t partition) const{
        return S_cnts[partition];
    }
};

struct chain_iterator_ref{
    vec4 x  ;
    int  cnt;
};

struct chain_iterator_i_ref{
    int32_t x;
    bool    v;
};

class chain_iterator{
private:
    const int32_t   * __restrict__ S_parts      ;
    const uint32_t  * __restrict__ S_chains     ;
    const uint32_t cnt ;
    
    const int32_t   * __restrict__ ptr          ;
    
    uint32_t current_bucket                     ;
    uint32_t next_bucket                        ;
    uint32_t i                                  ;
public:
    // __device__ __forceinline__ chain_iterator(
    //                 const int32_t   * __restrict__ S_parts      ,
    //                 const uint32_t  * __restrict__ S_cnts       ,
    //                 const uint32_t  * __restrict__ S_chains     ):
    //     S_parts(S_parts), S_chains(S_chains), cnt(S_cnts[blockIdx.x]), current_bucket(blockIdx.x), i(0){}

    __device__ __forceinline__ chain_iterator(
                    const int32_t   * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 current_partition):
        S_parts(S_parts + 4 * threadIdx.x), S_chains(S_chains), cnt((S_cnts[current_partition]/(4 * blockDim.x))*4 + max(((int32_t) (S_cnts[current_partition] % (4 * blockDim.x))) - ((int32_t) (4 * threadIdx.x)), 0)), ptr(S_parts + ((size_t) current_partition << log2_bucket_size) + 4 * threadIdx.x), current_bucket(current_partition), next_bucket(S_chains[current_partition]), i(0){}

    // __device__ __forceinline__ chain_iterator(
    //                 const uint32_t  * __restrict__ S_cnts):
    //    cnt(0), i(((S_cnts[blockIdx.x] + 4 * blockDim.x - 1)/(4 * blockDim.x)) * 4 * blockDim.x){}

    __device__ __forceinline__ chain_iterator(
                    const uint32_t  * __restrict__ S_cnts,
                          uint32_t                 current_partition):
       cnt(0), i(((S_cnts[current_partition] + 4 * blockDim.x - 1)/(4 * blockDim.x))*4){}

    __device__ __forceinline__ chain_iterator& operator++(){
        i   += 4;// * blockDim.x;
        ptr += 4 * blockDim.x;

        if ((i * blockDim.x) & bucket_size_mask) return *this;
        
        current_bucket = next_bucket;//int_shared[0];

        ptr = S_parts + (current_bucket << log2_bucket_size);

        next_bucket = S_chains[next_bucket];

        return *this;
    }

    __device__ __forceinline__ chain_iterator_ref operator*() const {
        chain_iterator_ref tmp;
        tmp.x   = *reinterpret_cast<const vec4 *>(ptr);
        tmp.cnt = cnt - i;
        return tmp;
    }

    __device__ __forceinline__ bool operator!=(const chain_iterator& o){
        return i != o.i;
    }
};

class chain_iterator_i{
private:
    const int32_t   * __restrict__ S_parts      ;
    const uint32_t  * __restrict__ S_chains     ;
    const uint32_t cnt ;
    
    const int32_t   * __restrict__ ptr          ;
    
    uint32_t current_bucket                     ;
    uint32_t next_bucket                        ;
    uint32_t i                                  ;
public:
    // __device__ __forceinline__ chain_iterator_i(
    //                 const int32_t   * __restrict__ S_parts      ,
    //                 const uint32_t  * __restrict__ S_cnts       ,
    //                 const uint32_t  * __restrict__ S_chains     ):
    //     S_parts(S_parts), S_chains(S_chains), cnt(S_cnts[blockIdx.x]), current_bucket(blockIdx.x), i(0){}

    __device__ __forceinline__ chain_iterator_i(
                    const int32_t   * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 current_partition):
        S_parts(S_parts + threadIdx.x), S_chains(S_chains), cnt((S_cnts[current_partition]/blockDim.x) + max(((int32_t) (S_cnts[current_partition] % (blockDim.x))) - ((int32_t) (threadIdx.x)), 0)), ptr(S_parts + ((size_t) current_partition << log2_bucket_size) + threadIdx.x), current_bucket(current_partition), next_bucket(S_chains[current_partition]), i(0){}

    // __device__ __forceinline__ chain_iterator_i(
    //                 const uint32_t  * __restrict__ S_cnts):
    //    cnt(0), i(((S_cnts[blockIdx.x] + 4 * blockDim.x - 1)/(4 * blockDim.x)) * 4 * blockDim.x){}

    __device__ __forceinline__ chain_iterator_i(
                    const uint32_t  * __restrict__ S_cnts,
                          uint32_t                 current_partition):
       cnt(0), i(((S_cnts[current_partition] + blockDim.x - 1)/(blockDim.x))){}

    __device__ __forceinline__ chain_iterator_i& operator++(){
        ++i;// * blockDim.x;
        ptr += blockDim.x;

        if ((i * blockDim.x) & bucket_size_mask) return *this;
        
        current_bucket = next_bucket;//int_shared[0];

        ptr = S_parts + (current_bucket << log2_bucket_size);

        next_bucket = S_chains[next_bucket];

        return *this;
    }

    __device__ __forceinline__ chain_iterator_i_ref operator*() const {
        chain_iterator_i_ref tmp;
        tmp.x = *ptr;
        tmp.v = i < cnt;
        return tmp;
    }

    __device__ __forceinline__ bool operator!=(const chain_iterator_i& o){
        return i != o.i;
    }
};

class chain_i{
private:
    const int32_t   * __restrict__ S_parts  ;
    const uint32_t  * __restrict__ S_cnts   ;
    const uint32_t  * __restrict__ S_chains ;
    const uint32_t                 partition;
public:
    __device__ __host__ __forceinline__ chain_i(
                    const int32_t   * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 partition):
        S_parts(S_parts), S_cnts(S_cnts), S_chains(S_chains), partition(partition){}

    __device__ __forceinline__ chain_iterator_i begin() const {
        return chain_iterator_i(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __forceinline__ chain_iterator_i end() const {
        return chain_iterator_i(S_cnts, partition);
    }
};

class chain{
private:
    const int32_t   * __restrict__ S_parts  ;
    const uint32_t  * __restrict__ S_cnts   ;
    const uint32_t  * __restrict__ S_chains ;
    const uint32_t                 partition;
public:
    __device__ __host__ __forceinline__ chain(
                    const int32_t   * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ,
                          uint32_t                 partition):
        S_parts(S_parts), S_cnts(S_cnts), S_chains(S_chains), partition(partition){}

    __device__ __forceinline__ chain_iterator begin() const {
        return chain_iterator(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __forceinline__ chain_iterator end() const {
        return chain_iterator(S_cnts, partition);
    }
};


class chains{
private:
    const int32_t   * __restrict__ S_parts  ;
    const uint32_t  * __restrict__ S_cnts   ;
    const uint32_t  * __restrict__ S_chains ;
public:
    __device__ __host__ __forceinline__ chains(
                    const int32_t   * __restrict__ S_parts      ,
                    const uint32_t  * __restrict__ S_cnts       ,
                    const uint32_t  * __restrict__ S_chains     ):
        S_parts(S_parts), S_cnts(S_cnts), S_chains(S_chains){}

    __device__ __host__ __forceinline__ chain get_chain(uint32_t partition) const{
        return chain(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __host__ __forceinline__ chain_i get_chain_i(uint32_t partition) const{
        return chain_i(S_parts, S_cnts, S_chains, partition);
    }

    __device__ __forceinline__ uint32_t get_chain_size(uint32_t partition) const{
        return S_cnts[partition];
    }
};

/*essentially the join_partitioned_aggregate*/
__global__ void join_partitioned_shared (
                                    const int32_t*               R,
                                    const int32_t*               Pr,
                                    const uint32_t*              R_cnts,
                                    const uint32_t*              R_chain,
                                    const int32_t*               S,
                                    const int32_t*               Ps,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    int32_t*                      results) {
    __shared__ int16_t elem[4096 + 512];
    __shared__ int32_t payload[4096 + 512];
    __shared__ int16_t next[4096 + 512];
    __shared__ int32_t head[LOCAL_BUCKETS];


    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    int lid = tid % 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int pr = -1;
    int ps = -1;


    for (uint32_t p = block; p < parts; p += pwidth) {
        int len_R = R_cnts[p];
        int len_S = S_cnts[p];

        if (len_S > 4096 + 512) {
            /*it was a microbenchmark so I didn't code this part*/
            continue;
        } else {
            chain R_chains(R, R_cnts, R_chain, p);
            chain Pr_chains(Pr, R_cnts, R_chain, p);

            chain S_chains(S, S_cnts, S_chain, p);
            chain Ps_chains(Ps, S_cnts, S_chain, p);

            int off = 0;

            for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                head[i] = -1;

            int rem_s = len_S % 4096;
            rem_s = (rem_s + 4 - 1)/4;

            __syncthreads();

            chain_iterator it_S = S_chains.begin();
            chain_iterator it_Ps = Ps_chains.begin();

            for (;it_S != S_chains.end(); ++it_S, ++it_Ps) {
                vec4 data_S = (*it_S).x;
                vec4 data_Ps = (*it_Ps).x;
                int l_cnt_S = (*it_S).cnt;
                
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    if (k < l_cnt_S) {
                        int val = data_S.i[k];
                        elem[off + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        payload[off + tid] = data_Ps.i[k];
                        int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                        int32_t last = atomicExch(&head[hval], off + tid);
                        next[off + tid] = last;
                    }   

                    off += (off < 4096)? blockDim.x : rem_s;
                }               
            }

            __syncthreads();


            chain_iterator it_R = R_chains.begin();
            chain_iterator it_Pr = Pr_chains.begin();

            for (;it_R != R_chains.end(); ++it_R, ++it_Pr) {
                vec4 data_R = (*it_R).x;
                vec4 data_Pr = (*it_Pr).x;
                int l_cnt_R = (*it_R).cnt;

                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int32_t val = data_R.i[k];
                    int32_t pval = data_Pr.i[k];
                    int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                    int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);

                    if (k < l_cnt_R) {
                        int32_t pos = head[hval];
                        while (pos >= 0) {
                            if (elem[pos] == tval) {
                                count += pval*payload[pos];
                            }

                            pos = next[pos];
                        }
                    }                   
                }
            }


            __syncthreads();
        }
    }

    atomicAdd(results, count);

    __syncthreads();
}

/*essentially the join_partitioned_aggregate but builds hashtable in GPU memory*/
__global__ void join_partitioned_global (
                                    const int32_t*               R,
                                    const int32_t*               Pr,
                                    const uint32_t*              R_cnts,
                                    const uint32_t*              R_chain,
                                    const int32_t*               S,
                                    const int32_t*               Ps,
                                    const uint32_t*              S_cnts,
                                    const uint32_t*              S_chain,
                                    int32_t                      log_parts,
                                    int32_t*                     results,
                                    int32_t*                     buffer) {
    
    int tid = threadIdx.x;
    int block = blockIdx.x;
    int width = blockDim.x;
    int pwidth = gridDim.x;
    int parts = 1 << log_parts;

    buffer += block*8*4096;

    int16_t* elem = (int16_t*) buffer;
    int32_t* payload = buffer + 4096 + 512;;
    int16_t* next = (int16_t*) (buffer + 2*(4096 + 512));
    int32_t* head = buffer + 3*(4096+512);

    

    int lid = tid % 32;
    int gnum = blockDim.x/32;

    int count = 0;

    int pr = -1;
    int ps = -1;


    for (uint32_t p = block; p < parts; p += pwidth) {
        chain R_chains(R, R_cnts, R_chain, p);
        chain Pr_chains(Pr, R_cnts, R_chain, p);

        chain S_chains(S, S_cnts, S_chain, p);
        chain Ps_chains(Ps, S_cnts, S_chain, p);

        int len_R = R_cnts[p];
        int len_S = S_cnts[p];

        if (len_S > 4096 + 512) {
           /*it was a microbenchmark so I didn't code this part*/
            continue;
        } else {
            int off = 0;

            for (int i = tid; i < LOCAL_BUCKETS; i += blockDim.x)
                head[i] = -1;

            int rem_s = len_S % 4096;
            rem_s = (rem_s + 4 - 1)/4;

            __syncthreads();

            chain_iterator it_S = S_chains.begin();
            chain_iterator it_Ps = Ps_chains.begin();

            for (;it_S != S_chains.end(); ++it_S, ++it_Ps) {
                vec4 data_S = (*it_S).x;
                vec4 data_Ps = (*it_Ps).x;
                int l_cnt_S = (*it_S).cnt;
                
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    if (k < l_cnt_S) {
                        int val = data_S.i[k];
                        elem[off + tid] = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                        payload[off + tid] = data_Ps.i[k];
                        int hval = (val >> log_parts) & (LOCAL_BUCKETS - 1);

                        int32_t last = atomicExch(&head[hval], off + tid);
                        next[off + tid] = last;
                    }   

                    off += (off < 4096)? blockDim.x : rem_s;
                }               
            }

            __syncthreads();

            chain_iterator it_R = R_chains.begin();
            chain_iterator it_Pr = Pr_chains.begin();

            for (;it_R != R_chains.end(); ++it_R, ++it_Pr) {
                vec4 data_R = (*it_R).x;
                vec4 data_Pr = (*it_Pr).x;
                int l_cnt_R = (*it_R).cnt;

                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    int32_t val = data_R.i[k];
                    int32_t pval = data_Pr.i[k];
                    int16_t tval = (int16_t) (val >> (LOCAL_BUCKETS_BITS + log_parts));
                    int32_t hval =  (val >> log_parts) & (LOCAL_BUCKETS - 1);

                    if (k < l_cnt_R) {
                        int32_t pos = head[hval];
                        while (pos >= 0) {
                            if (elem[pos] == tval) {
                                count += pval*payload[pos];
                            }

                            pos = next[pos];
                        }
                    }                   
                }
            }

            __syncthreads();
        }
    }

    atomicAdd(results, count);

    __syncthreads();
}