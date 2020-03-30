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

#include <algorithm>
#include <cassert>
#include <cinttypes>

#include "common.h"
#include "common-host.h"

#include "join-primitives.cuh"
#include "partition-primitives.cuh"

#include <iostream>
#include <numa.h>
#include <unistd.h>

#include <list>

#include <omp.h>
#include <immintrin.h>


#define PROBE_BUCKETS 256
#define LOCAL_BUCKETS 2048 


#define UVA_OPTION3_

unsigned int outOfGPU_Join1_payload_uva (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, timingInfo *time, uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    first_bit = 0;

    size_t buckets_num_max_R    = ((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S    = ((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int* R_gpu;
    int* S_gpu;

    int* R_gpu_final;
    int* S_gpu_final;

    int* R_gpu_temp;
    int* S_gpu_temp;

    int* Pr_gpu;
    int* Ps_gpu;

    int* Pr_gpu_final;
    int* Ps_gpu_final;

    int* Pr_gpu_temp;
    int* Ps_gpu_temp;

    int p = 27;

    #ifdef UVA_OPTION1_
    printf ("opt1\n");
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu, R_gpu, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu, Pr_gpu, 0);
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaHostAlloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu, S_gpu, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu, Ps_gpu, 0);
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    #else  
    #ifdef UVA_OPTION2_
    printf ("opt2\n");
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu, R_gpu, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu, Pr_gpu, 0);
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu_temp, R_gpu_temp, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu_temp, Pr_gpu_temp, 0);
    CHK_ERROR(cudaMalloc((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaHostAlloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu, S_gpu, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu, Ps_gpu, 0);
    CHK_ERROR(cudaHostAlloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu_temp, S_gpu_temp, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu_temp, Ps_gpu_temp, 0);
    CHK_ERROR(cudaMalloc((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    #else
    #ifdef UVA_OPTION3_
    printf ("opt3\n");
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu, R_gpu, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu, Pr_gpu, 0);
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu_temp, R_gpu_temp, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu_temp, Pr_gpu_temp, 0);
    CHK_ERROR(cudaHostAlloc((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &R_gpu_final, R_gpu_final, 0);
    cudaHostGetDevicePointer((void**) &Pr_gpu_final, Pr_gpu_final, 0);

    CHK_ERROR(cudaHostAlloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu, S_gpu, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu, Ps_gpu, 0);
    CHK_ERROR(cudaHostAlloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu_temp, S_gpu_temp, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu_temp, Ps_gpu_temp, 0);
    CHK_ERROR(cudaHostAlloc((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    CHK_ERROR(cudaHostAlloc((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &S_gpu_final, S_gpu_final, 0);
    cudaHostGetDevicePointer((void**) &Ps_gpu_final, Ps_gpu_final, 0);
    #else
    printf ("else\n");
    CHK_ERROR(cudaMalloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    R_gpu_final = R_gpu;
    Pr_gpu_final = Pr_gpu;

    CHK_ERROR(cudaMalloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    S_gpu_final = S_gpu;
    Ps_gpu_final = Ps_gpu;
    #endif
    #endif
    #endif

    int32_t* output;
    CHK_ERROR(cudaHostAlloc((void**) &output, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &output, output, 0);

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));

    uint32_t* chains_R[2];
    uint32_t* chains_S[2];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[2];

    uint64_t* heads_R[2];
    uint64_t* heads_S[2];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[2];

    int* aggr_cnt;

    CHK_ERROR(cudaMalloc((void**) &aggr_cnt, 64*sizeof(int)));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));

    for (int i = 0; i < 2; i++) {
        CHK_ERROR(cudaMalloc((void**) &chains_R[i], buckets_num_max_R * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_R[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_R[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_R[i], sizeof(uint32_t)));

        CHK_ERROR(cudaMalloc((void**) &chains_S[i], buckets_num_max_S * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_S[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_S[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_S[i], sizeof(uint32_t)));
    }

    cudaMemcpy(R_gpu, R, RelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(Pr_gpu, Pr, RelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(S_gpu, S, SelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(Ps_gpu, Ps, SelsNum*sizeof(int), cudaMemcpyDefault);

    CHK_ERROR(cudaDeviceSynchronize());

    double t1 = cpuSeconds();

    prepare_Relation_payload_triple (  R_gpu, R_gpu_temp, R_gpu_final,
                                Pr_gpu, Pr_gpu_temp, Pr_gpu_final,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload_triple (  S_gpu, S_gpu_temp, S_gpu_final,
                                Ps_gpu, Ps_gpu_temp, Ps_gpu_final,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize()); 

    double t3 = cpuSeconds();
  
    uint32_t* bucket_info_R = (uint32_t*) Pr_gpu_temp;
    

    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);
        
    join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R,
                                    S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);    

    CHK_ERROR(cudaDeviceSynchronize());
    double t2 = cpuSeconds();

    cudaDeviceSynchronize();

    std::cout << "Without materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    CHK_ERROR(cudaDeviceSynchronize());

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));
    
    cudaMemcpy(R_gpu, R, RelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(Pr_gpu, Pr, RelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(S_gpu, S, SelsNum*sizeof(int), cudaMemcpyDefault);
    cudaMemcpy(Ps_gpu, Ps, SelsNum*sizeof(int), cudaMemcpyDefault);

    CHK_ERROR(cudaDeviceSynchronize());

    t1 = cpuSeconds();

    prepare_Relation_payload_triple (  R_gpu, R_gpu_temp, R_gpu_final,
                                Pr_gpu, Pr_gpu_temp, Pr_gpu_final,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload_triple (  S_gpu, S_gpu_temp, S_gpu_final,
                                Ps_gpu, Ps_gpu_temp, Ps_gpu_final,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize());

    t3 = cpuSeconds();
 
    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);

    join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R, 
                                    S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);
        
    CHK_ERROR(cudaDeviceSynchronize());
    t2 = cpuSeconds();

    cudaDeviceSynchronize();

    std::cout << "Without materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    return 1;       
}

#define UM_OPTION1_ 

unsigned int outOfGPU_Join1_payload_um (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, timingInfo *time, uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    first_bit = 0;

    size_t buckets_num_max_R    = ((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S    = ((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int* R_gpu;
    int* S_gpu;

    int* R_gpu_final;
    int* S_gpu_final;

    int* R_gpu_temp;
    int* S_gpu_temp;

    int* Pr_gpu;
    int* Ps_gpu;

    int* Pr_gpu_final;
    int* Ps_gpu_final;

    int* Pr_gpu_temp;
    int* Ps_gpu_temp;

    int p = 27;

    #ifdef UM_OPTION1_
    printf ("opt1\n");
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMallocManaged((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    #else  
    #ifdef UM_OPTION2_
    printf ("opt2\n");
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMallocManaged((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    #else
    #ifdef UM_OPTION3_
    printf ("opt3\n");
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &R_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Pr_gpu_final, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMallocManaged((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &S_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMallocManaged((void**) &Ps_gpu_final, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    #else
    printf ("else\n");
    CHK_ERROR(cudaMalloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    R_gpu_final = R_gpu;
    Pr_gpu_final = Pr_gpu;

    CHK_ERROR(cudaMalloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    S_gpu_final = S_gpu;
    Ps_gpu_final = Ps_gpu;
    #endif
    #endif
    #endif

    int32_t* output;
    CHK_ERROR(cudaHostAlloc((void**) &output, buckets_num_max_S * bucket_size * sizeof(int32_t), cudaHostAllocMapped));
    cudaHostGetDevicePointer((void**) &output, output, 0);

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));

    uint32_t* chains_R[2];
    uint32_t* chains_S[2];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[2];

    uint64_t* heads_R[2];
    uint64_t* heads_S[2];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[2];

    int* aggr_cnt;

    CHK_ERROR(cudaMalloc((void**) &aggr_cnt, 64*sizeof(int)));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));

    for (int i = 0; i < 2; i++) {
        CHK_ERROR(cudaMalloc((void**) &chains_R[i], buckets_num_max_R * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_R[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_R[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_R[i], sizeof(uint32_t)));

        CHK_ERROR(cudaMalloc((void**) &chains_S[i], buckets_num_max_S * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_S[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_S[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_S[i], sizeof(uint32_t)));
    }

    memcpy(R_gpu, R, RelsNum*sizeof(int));
    memcpy(Pr_gpu, Pr, RelsNum*sizeof(int));
    memcpy(S_gpu, S, SelsNum*sizeof(int));
    memcpy(Ps_gpu, Ps, SelsNum*sizeof(int));

    CHK_ERROR(cudaDeviceSynchronize());

    double t1 = cpuSeconds();

    prepare_Relation_payload_triple (  R_gpu, R_gpu_temp, R_gpu_final,
                                Pr_gpu, Pr_gpu_temp, Pr_gpu_final,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload_triple (  S_gpu, S_gpu_temp, S_gpu_final,
                                Ps_gpu, Ps_gpu_temp, Ps_gpu_final,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize()); 

    double t3 = cpuSeconds();
  
    uint32_t* bucket_info_R = (uint32_t*) Pr_gpu_temp;
    

    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);
        
    join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R,
                                    S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);    

    CHK_ERROR(cudaDeviceSynchronize());
    double t2 = cpuSeconds();

    cudaDeviceSynchronize();

    std::cout << "With materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    CHK_ERROR(cudaDeviceSynchronize());

    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));
    
    memcpy(R_gpu, R, RelsNum*sizeof(int));
    memcpy(Pr_gpu, Pr, RelsNum*sizeof(int));
    memcpy(S_gpu, S, SelsNum*sizeof(int));
    memcpy(Ps_gpu, Ps, SelsNum*sizeof(int));

    CHK_ERROR(cudaDeviceSynchronize());

    t1 = cpuSeconds();

    prepare_Relation_payload_triple (  R_gpu, R_gpu_temp, R_gpu_final,
                                Pr_gpu, Pr_gpu_temp, Pr_gpu_final,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload_triple (  S_gpu, S_gpu_temp, S_gpu_final,
                                Ps_gpu, Ps_gpu_temp, Ps_gpu_final,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize());

    t3 = cpuSeconds();
 
    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);

    join_partitioned_results <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu_final, Pr_gpu_final, chains_R[1], bucket_info_R, 
                                    S_gpu_final, Ps_gpu_final, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0], output);
        
    CHK_ERROR(cudaDeviceSynchronize());
    t2 = cpuSeconds();

    cudaDeviceSynchronize();

    std::cout << "Without materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    return 1;       
}



unsigned int outOfGPU_Join_payload_var (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, 
                                        timingInfo *time, int col_num1, int col_num2,
                                        uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    printf ("RelsNum = %d SelsNum = %d\n", RelsNum, SelsNum);

    first_bit = 0;

    size_t buckets_num_max_R    = ((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S    = ((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int* R_gpu;
    int* S_gpu;

    int* R_gpu_temp;
    int* S_gpu_temp;

    int* Pr_gpu;
    int* Ps_gpu;

    int* Pr_gpu_temp;
    int* Ps_gpu_temp;

    int* Dr_gpu;
    int* Ds_gpu;

    int p = 27;

    CHK_ERROR(cudaMalloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Dr_gpu, col_num1 * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMalloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ds_gpu, col_num2 * buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));

    uint32_t* chains_R[2];
    uint32_t* chains_S[2];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[2];

    uint64_t* heads_R[2];
    uint64_t* heads_S[2];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[2];

    int* aggr_cnt;

    CHK_ERROR(cudaMalloc((void**) &aggr_cnt, 64*sizeof(int)));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));

    for (int i = 0; i < 2; i++) {
        CHK_ERROR(cudaMalloc((void**) &chains_R[i], buckets_num_max_R * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_R[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_R[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_R[i], sizeof(uint32_t)));

        CHK_ERROR(cudaMalloc((void**) &chains_S[i], buckets_num_max_S * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_S[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_S[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_S[i], sizeof(uint32_t)));
    }

    cudaMemcpy(R_gpu, R, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Dr_gpu, Pr, col_num1*RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_gpu, S, SelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ds_gpu, Ps, col_num2*SelsNum*sizeof(int), cudaMemcpyHostToDevice);

    uint32_t* bucket_info_R = (uint32_t*) Pr_gpu_temp;

    CHK_ERROR(cudaDeviceSynchronize());

    double t1 = cpuSeconds();

    init_payload <<<64, 1024, 0, streams_R[1]>>> (Pr_gpu, RelsNum);
    init_payload <<<64, 1024, 0, streams_R[1]>>> (Ps_gpu, SelsNum);

    prepare_Relation_payload (  R_gpu, R_gpu_temp, 
                                Pr_gpu, Pr_gpu_temp,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload (  S_gpu, S_gpu_temp, 
                                Ps_gpu, Ps_gpu_temp,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize()); 

    double t3 = cpuSeconds();
 

    int32_t* output = (int32_t*) R_gpu_temp;

    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);
 
    
    join_partitioned_varpayload <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (
                                    R_gpu, Pr_gpu, Dr_gpu, chains_R[1], bucket_info_R,
                                    S_gpu, Ps_gpu, Ds_gpu, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, col_num1, col_num2, RelsNum, buckets_used_R[1], &aggr_cnt[0]);
                                    
        
    CHK_ERROR(cudaDeviceSynchronize());
    double t2 = cpuSeconds();

    cudaDeviceSynchronize();

    cudaFree((void*) R_gpu);
    cudaFree((void*) Pr_gpu);
    cudaFree((void*) Dr_gpu);

    cudaFree((void*) S_gpu);
    cudaFree((void*) Ps_gpu);
    cudaFree((void*) Ds_gpu);

    cudaFree((void*) R_gpu_temp);
    cudaFree((void*) Pr_gpu_temp);

    cudaFree((void*) S_gpu_temp);
    cudaFree((void*) Ps_gpu_temp);

    cudaFree((void*) chains_R[0]);
    cudaFree((void*) chains_R[1]);

    cudaFree((void*) chains_S[0]);
    cudaFree((void*) chains_S[1]);

    cudaFree((void*) cnts_R[0]);
    cudaFree((void*) cnts_R[1]);

    cudaFree((void*) cnts_S[0]);
    cudaFree((void*) cnts_S[1]);

    cudaFree((void*) heads_R[0]);
    cudaFree((void*) heads_R[1]);

    cudaFree((void*) heads_S[0]);
    cudaFree((void*) heads_S[1]);


    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;
    
    return 1;       
}

unsigned int outOfGPU_Join_payload_var2 (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, 
                                        timingInfo *time, int col_num1, int col_num2,
                                        uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    first_bit = 0;

    size_t buckets_num_max_R    = ((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S    = ((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int* R_gpu;
    int* S_gpu;

    int* R_gpu_temp;

    int* Pr_gpu;
    int* Ps_gpu;

    int* Dr_gpu;
    int* Ds_gpu;

    int p = 27;

    CHK_ERROR(cudaMalloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Dr_gpu, col_num1 * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, 2*buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMalloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ds_gpu, col_num2 * buckets_num_max_S * bucket_size * sizeof(int32_t)));

    cudaMemset (R_gpu_temp, 0, 2*buckets_num_max_R * bucket_size * sizeof(int32_t));

    int* aggr_cnt;

    CHK_ERROR(cudaMalloc((void**) &aggr_cnt, 64*sizeof(int)));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));

    cudaMemcpy(R_gpu, R, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Dr_gpu, Pr, col_num1*RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_gpu, S, SelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ds_gpu, Ps, col_num2*SelsNum*sizeof(int), cudaMemcpyHostToDevice);

    CHK_ERROR(cudaDeviceSynchronize());

    double t1 = cpuSeconds();

    init_payload <<<64, 1024, 0, streams_R[1]>>> (Ps_gpu, SelsNum);

    build_perfect_array <<<64, 1024, 0, streams_R[1]>>> (S_gpu, Ps_gpu, SelsNum, R_gpu_temp);

    CHK_ERROR(cudaDeviceSynchronize()); 

    double t3 = cpuSeconds();
 
    
    probe_perfect_array_varpay <<<64, 1024, 0, streams_R[1]>>> (R_gpu, Dr_gpu, RelsNum, R_gpu_temp, Ds_gpu, col_num1, col_num2, RelsNum, aggr_cnt);
                               
        
    CHK_ERROR(cudaDeviceSynchronize());
    double t2 = cpuSeconds();

    cudaDeviceSynchronize();

    cudaFree((void*) R_gpu);
    cudaFree((void*) Pr_gpu);
    cudaFree((void*) Dr_gpu);

    cudaFree((void*) S_gpu);
    cudaFree((void*) Ps_gpu);
    cudaFree((void*) Ds_gpu);

    cudaFree((void*) R_gpu_temp);

    std::cout << "Results for " << col_num1 << " and " << col_num2 << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    
    return 1;       
}

unsigned int outOfGPU_Join1_payload (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, timingInfo *time, uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    first_bit = 0;

    size_t buckets_num_max_R    = 2*((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;
    size_t buckets_num_max_S    = 2*((((SelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int* R_gpu;
    int* S_gpu;

    int* R_gpu_temp;
    int* S_gpu_temp;

    int* Pr_gpu;
    int* Ps_gpu;

    int* Pr_gpu_temp;
    int* Ps_gpu_temp;

    int p = 27;

    CHK_ERROR(cudaMalloc((void**) &R_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu, buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &R_gpu_temp, 2*buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_gpu_temp, buckets_num_max_R * bucket_size * sizeof(int32_t)));

    CHK_ERROR(cudaMalloc((void**) &S_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &S_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_gpu_temp, buckets_num_max_S * bucket_size * sizeof(int32_t)));

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));

    uint32_t* chains_R[2];
    uint32_t* chains_S[2];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[2];

    uint64_t* heads_R[2];
    uint64_t* heads_S[2];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[2];

    int* aggr_cnt;

    CHK_ERROR(cudaMalloc((void**) &aggr_cnt, 64*sizeof(int)));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));

    for (int i = 0; i < 2; i++) {
        CHK_ERROR(cudaMalloc((void**) &chains_R[i], buckets_num_max_R * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_R[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_R[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_R[i], sizeof(uint32_t)));

        CHK_ERROR(cudaMalloc((void**) &chains_S[i], buckets_num_max_S * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &cnts_S[i], parts2 * sizeof(uint32_t)));
        CHK_ERROR(cudaMalloc((void**) &heads_S[i], parts2 * sizeof(uint64_t)));
        CHK_ERROR(cudaMalloc((void**) &buckets_used_S[i], sizeof(uint32_t)));
    }

    cudaMemcpy(R_gpu, R, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Pr_gpu, Pr, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(S_gpu, S, SelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(Ps_gpu, Ps, SelsNum*sizeof(int), cudaMemcpyHostToDevice);

    CHK_ERROR(cudaDeviceSynchronize());

    double t1 = cpuSeconds();

    prepare_Relation_payload (  R_gpu, R_gpu_temp, 
                                Pr_gpu, Pr_gpu_temp,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload (  S_gpu, S_gpu_temp, 
                                Ps_gpu, Ps_gpu_temp,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize()); 

    double t3 = cpuSeconds();
 

 
    uint32_t* bucket_info_R = (uint32_t*) Pr_gpu_temp;
    int32_t* output = (int32_t*) R_gpu_temp;

    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);
 
    /*
    join_partitioned_aggregate <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (
                                    R_gpu, Pr_gpu, chains_R[1], bucket_info_R,
                                    S_gpu, Ps_gpu, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);
                                    */
        
    join_partitioned_results <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu, Pr_gpu, chains_R[1], bucket_info_R,
                                    S_gpu, Ps_gpu, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0], output);    

    //build_perfect_array <<<64, 1024>>> (R_gpu, Pr_gpu, RelsNum, R_gpu_temp);
    //probe_perfect_array <<<64, 1024>>> (S_gpu, Ps_gpu, SelsNum, R_gpu_temp, aggr_cnt);

    //buildBucketChaining <<<64, 1024>>> (R_gpu, R_gpu_temp, Pr_gpu_temp, p, RelsNum);
    //probeBucketChaining <<<64, 1024>>> (S_gpu, R_gpu, Pr_gpu, R_gpu_temp, Pr_gpu_temp, p, SelsNum);
 
    //build_ht_chains <<<64, 1024>>> (R_gpu, RelsNum, p, R_gpu_temp, Pr_gpu_temp);
    //chains_probing <<<64, 1024>>> (S_gpu, Ps_gpu, SelsNum, p, R_gpu_temp, R_gpu, Pr_gpu, Pr_gpu_temp, aggr_cnt);

    //ht_hist <<<64, 1024>>> (R_gpu, RelsNum, p, S_gpu_temp);
    //ht_offsets <<<64, 1024>>> (p, S_gpu_temp, S_gpu_temp + (1 << (p+2)), aggr_cnt+2);
    //build_ht_linear <<<64, 1024>>>(R_gpu, Pr_gpu, RelsNum, p, S_gpu_temp + (1 << (p+2)), R_gpu_temp, Pr_gpu_temp);
    //linear_probing <<<64, 1024>>> (S_gpu, Ps_gpu, R_gpu_temp, Pr_gpu_temp, S_gpu_temp, S_gpu_temp + (1 << (p+2)), SelsNum, p, aggr_cnt);

    CHK_ERROR(cudaDeviceSynchronize());
    double t2 = cpuSeconds();

    cudaDeviceSynchronize();

    std::cout << "With materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput  " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    CHK_ERROR(cudaDeviceSynchronize());

    cudaMemset (R_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (Pr_gpu_temp, 0, buckets_num_max_R * bucket_size * sizeof(int32_t));
    cudaMemset (aggr_cnt, 0, 64 * sizeof(int32_t));
    cudaMemcpy (R_gpu, R, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (Pr_gpu, Pr, RelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (S_gpu, S, SelsNum*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (Ps_gpu, Ps, SelsNum*sizeof(int), cudaMemcpyHostToDevice);

    CHK_ERROR(cudaDeviceSynchronize());

    t1 = cpuSeconds();

    prepare_Relation_payload (  R_gpu, R_gpu_temp, 
                                Pr_gpu, Pr_gpu_temp,
                                RelsNum,
                                buckets_num_max_R, heads_R, cnts_R, chains_R, buckets_used_R, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    prepare_Relation_payload (  S_gpu, S_gpu_temp, 
                                Ps_gpu, Ps_gpu_temp,
                                SelsNum,
                                buckets_num_max_S, heads_S, cnts_S, chains_S, buckets_used_S, 
                                log_parts1, log_parts2, first_bit, streams_R[1], NULL, OMP_PARALLELISM1);

    CHK_ERROR(cudaDeviceSynchronize());

    t3 = cpuSeconds();
 
    decompose_chains <<<(1 << log_parts1), 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[1], cnts_R[1], log_parts1 + log_parts2, 2*bucket_size);

    join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu, Pr_gpu, chains_R[1], bucket_info_R, 
                                    S_gpu, Ps_gpu, cnts_S[1], chains_S[1],
                                    log_parts1 + log_parts2, buckets_used_R[1], &aggr_cnt[0]);
        
    CHK_ERROR(cudaDeviceSynchronize());
    t2 = cpuSeconds();

    cudaDeviceSynchronize();

    int results;
    cudaMemcpy(&results, aggr_cnt, sizeof(int), cudaMemcpyDeviceToHost);
    printf ("%d results\n", results);

    std::cout << "Without materialization" << std::endl;
    std::cout << "Partition Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;
    std::cout << "Joins Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t3)/1000/1000 << std::endl;
    std::cout << "Total Throughput " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t2-t1)/1000/1000 << std::endl;

    return 1;       
}

/*out-of-gpu case*/



unsigned int outOfGPU_Join2_payload (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, timingInfo *time, uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    cudaSetDevice(1);

    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t log_parts_host = 4;
    uint32_t parts_host = 1 << log_parts_host;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    printf ("RelsNum = %d SelsNum = %d\n", RelsNum, SelsNum);

    size_t R_part_size = RelsNum / parts_host;
    size_t S_segment_size = (RelsNum <= 2*CHUNK_SIZE) ? RelsNum / 4 : CHUNK_SIZE;
    size_t S_segment_num = (SelsNum + S_segment_size -1)/S_segment_size;
    size_t S_part_size = S_segment_size / parts_host;

    size_t buckets_num_max_R    = ((((R_part_size + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + parts2/4; 
    size_t buckets_num_max_S    = ((((S_part_size + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;

    int32_t * aggr_cnt;
    CHK_ERROR(cudaMalloc((void**) &aggr_cnt,    1024 * parts_host * S_segment_num * sizeof(int32_t)));
    CHK_ERROR(cudaMemset(aggr_cnt, 0,           1024 * parts_host * S_segment_num * sizeof(int32_t)));

    
    int* R_gpu[PARTS_RESIDENT+1];
    int* S_gpu[PARTS_RESIDENT+1];

    int* Pr_gpu[PARTS_RESIDENT+1];
    int* Ps_gpu[PARTS_RESIDENT+1];
    
    uint32_t* chains_R[PARTS_RESIDENT+1];
    uint32_t* chains_S[PARTS_RESIDENT+1];

    uint32_t* cnts_R[PARTS_RESIDENT+1];
    uint32_t* cnts_S[PARTS_RESIDENT+1];

    uint64_t* heads_R[PARTS_RESIDENT+1];
    uint64_t* heads_S[PARTS_RESIDENT+1];

    uint32_t* buckets_used_R[PARTS_RESIDENT+1];
    uint32_t* buckets_used_S[PARTS_RESIDENT+1];

    size_t*   offsets_R[16];
    size_t*   offsets_S[16][64];
    
    
    int* output_host;

    int* R_buffer;
    int* Pr_buffer;
    uint32_t* chains_R_buffer;
    uint32_t* cnts_R_buffer;
    uint64_t* heads_R_buffer;
    uint32_t* buckets_used_R_buffer;
    size_t*   offsets_R_buffer;

    uint32_t* bucket_info_R[PARTS_RESIDENT];

    CHK_ERROR(cudaMalloc((void**) &R_buffer,                (PARTS_RESIDENT+1) * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_buffer,               (PARTS_RESIDENT+1) * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &chains_R_buffer,         (PARTS_RESIDENT+1) * buckets_num_max_R * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &cnts_R_buffer,           (PARTS_RESIDENT+1) * parts2 * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &heads_R_buffer,          (PARTS_RESIDENT+1) * parts2 * sizeof(uint64_t)));
    CHK_ERROR(cudaMalloc((void**) &buckets_used_R_buffer,   (PARTS_RESIDENT+1) * sizeof(uint32_t)));


    for (int i = 0; i < PARTS_RESIDENT; i++)
        CHK_ERROR(cudaMalloc((void**) &bucket_info_R[i],           buckets_num_max_R * sizeof(uint32_t)));



    CHK_ERROR(cudaMalloc((void**) &offsets_R_buffer,        parts_host * OMP_PARALLELISM1 * 4 * sizeof(size_t)));

    for (int i = 0; i < PARTS_RESIDENT+1; i++) {
        R_gpu[i]        = R_buffer + i * buckets_num_max_R * bucket_size;
        Pr_gpu[i]       = Pr_buffer + i * buckets_num_max_R * bucket_size;
        chains_R[i]     = chains_R_buffer + i * buckets_num_max_R;
        cnts_R[i]       = cnts_R_buffer + i * parts2;
        heads_R[i]      = heads_R_buffer + i * parts2;
        buckets_used_R[i] = buckets_used_R_buffer + i;
    }

    for (int i = 0; i < parts_host; i++)
        offsets_R[i]    = offsets_R_buffer + i * OMP_PARALLELISM1 * 4;

    cudaMemset (R_buffer, 0, (PARTS_RESIDENT+1) * buckets_num_max_R * bucket_size * sizeof(int32_t));

    int* S_buffer;
    int* Ps_buffer;
    uint32_t* chains_S_buffer;
    uint32_t* cnts_S_buffer;
    uint64_t* heads_S_buffer;
    uint32_t* buckets_used_S_buffer;

    size_t* offsets_S_buffer;

    CHK_ERROR(cudaMalloc((void**) &S_buffer,                3 * buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_buffer,               3 * buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &chains_S_buffer,         3 * buckets_num_max_S * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &cnts_S_buffer,           3 * parts2 * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &heads_S_buffer,          3 * parts2 * sizeof(uint64_t)));
    CHK_ERROR(cudaMalloc((void**) &buckets_used_S_buffer,   3 * sizeof(uint32_t)));

    CHK_ERROR(cudaMalloc((void**) &offsets_S_buffer,        S_segment_num * parts_host * OMP_PARALLELISM2 * 4 * sizeof(size_t)));

    for (int i = 0; i < 3; i++) {
        S_gpu[i]        = S_buffer + i * buckets_num_max_S * bucket_size;
        Ps_gpu[i]       = Ps_buffer + i * buckets_num_max_S * bucket_size;
        chains_S[i]     = chains_S_buffer + i * buckets_num_max_S;
        cnts_S[i]       = cnts_S_buffer + i * parts2;
        heads_S[i]      = heads_S_buffer + i * parts2;
        buckets_used_S[i] = buckets_used_S_buffer + i;
    }

    int32_t* output[2];

    output[0] = R_gpu[PARTS_RESIDENT];
    output[1] = Pr_gpu[PARTS_RESIDENT];

    for (int i = 0; i < parts_host; i++) {
        for (int j = 0; j < S_segment_num; j++)
            offsets_S[i][j] = offsets_S_buffer + (j*parts_host + i) * OMP_PARALLELISM2 * 4;
    }

    cudaMemset (S_buffer, 0, 3 * buckets_num_max_S * bucket_size * sizeof(int32_t));

    const uint64_t mask = 0x00000000000000FF;

    int* R_socket[2];
    int* Pr_socket[2];
    size_t* R_offsets_socket[2];
    int* R_output_socket[2];
    int* Pr_output_socket[2];

    size_t* R_offsets_virt_socket[2];

    for (int i = 0; i < 2; i++) {
        R_socket[i] = (int*) numa_alloc_onnode((RelsNum+1024)*sizeof(int), numa_node_of_cpu(i));
        Pr_socket[i] = (int*) numa_alloc_onnode((RelsNum+1024)*sizeof(int), numa_node_of_cpu(i));
        
        R_output_socket[i] = (int*) numa_alloc_onnode(2*(RelsNum + RelsNum/8)*sizeof(int), numa_node_of_cpu(i));
        Pr_output_socket[i] = R_output_socket[i] + (RelsNum + RelsNum/8);

        R_offsets_socket[i] = (size_t*) numa_alloc_onnode((OMP_PARALLELISM1+1)*OMP_MEMORY_STEP*sizeof(size_t), numa_node_of_cpu(i));
        R_offsets_virt_socket[i] = (size_t*) numa_alloc_onnode((OMP_PARALLELISM1+1)*OMP_MEMORY_STEP*sizeof(size_t), numa_node_of_cpu(i));

        while ((((uint64_t) R_socket[i]) & mask) != 0)
            (R_socket[i])++;

        while ((((uint64_t)  R_output_socket[i]) & mask) != 0)
            (R_output_socket[i])++;

        while ((((uint64_t) Pr_socket[i]) & mask) != 0)
            (Pr_socket[i])++;

        while ((((uint64_t)  Pr_output_socket[i]) & mask) != 0)
            (Pr_output_socket[i])++;

        while ((((uint64_t) R_offsets_socket[i]) & mask) != 0)
            (R_offsets_socket[i])++;

        memset(R_offsets_socket[i], 0, OMP_PARALLELISM1*OMP_MEMORY_STEP*sizeof(size_t));
        
        if (i == 1)
            cudaHostRegister(R_output_socket[i], 2 * (RelsNum + RelsNum/8) * sizeof(int), cudaHostRegisterMapped);
    }

    size_t total_R[2];
    size_t total_S[64][2];

    int* S_socket[2];
    int* Ps_socket[2];

    size_t* S_offsets_socket[2];

    int* S_output_socket[2];
    int* Ps_output_socket[2];

    int* S_segment_socket[64][2];
    int* Ps_segment_socket[64][2];

    size_t* S_segment_offsets_socket[64][2];

    int* S_segment_output_socket[64][2];
    int* Ps_segment_output_socket[64][2];

    size_t* S_segment_offsets_virt_socket[64][2];


    for (int i = 0; i < 2; i++) {
        S_socket[i] = (int*) numa_alloc_onnode((SelsNum + 4096)*sizeof(int), numa_node_of_cpu(i));
        Ps_socket[i] = (int*) numa_alloc_onnode((SelsNum + 4096)*sizeof(int), numa_node_of_cpu(i));

        S_output_socket[i] = (int*) numa_alloc_onnode(2*((SelsNum + SelsNum/8) + 4096)*sizeof(int), numa_node_of_cpu(i));
        Ps_output_socket[i] = S_output_socket[i] + ((SelsNum + SelsNum/8) + 4096);

        S_offsets_socket[i] = (size_t*) numa_alloc_onnode(S_segment_num*(OMP_PARALLELISM2+1)*OMP_MEMORY_STEP*sizeof(size_t), numa_node_of_cpu(i));

        const uint64_t mask = 0x00000000000000FF;

        while ((((uint64_t) S_socket[i]) & mask) != 0)
            (S_socket[i])++;

        while ((((uint64_t) S_output_socket[i]) & mask) != 0)
            (S_output_socket[i])++;

        while ((((uint64_t) Ps_socket[i]) & mask) != 0)
            (Ps_socket[i])++;

        while ((((uint64_t) Ps_output_socket[i]) & mask) != 0)
            (Ps_output_socket[i])++;

        while ((((uint64_t) S_offsets_socket[i]) & mask) != 0)
            (S_offsets_socket[i])++;

        memset(S_offsets_socket[i], 0, S_segment_num*OMP_PARALLELISM2*OMP_MEMORY_STEP*sizeof(size_t));
    
        for (int j = 0; j < S_segment_num; j++) {
            S_segment_socket[j][i]             = S_socket[i] + j * S_segment_size;
            Ps_segment_socket[j][i]            = Ps_socket[i] + j * S_segment_size;

            S_segment_offsets_socket[j][i]     = S_offsets_socket[i] + j * OMP_PARALLELISM2*OMP_MEMORY_STEP;

            S_segment_output_socket[j][i]      = S_output_socket[i] + j * (S_segment_size + S_segment_size/8);
            Ps_segment_output_socket[j][i]     = Ps_output_socket[i] + j * (S_segment_size + S_segment_size/8);

            if ((((uint64_t) S_segment_output_socket[j][i]) & 0x00000000000000FF) != 0)
                        printf ("erroorrrrrrrrrr\n");

            S_segment_offsets_virt_socket[j][i] = (size_t*) numa_alloc_onnode((OMP_PARALLELISM2+1)*OMP_MEMORY_STEP*sizeof(size_t), numa_node_of_cpu(i));
        }

        if (i == 1)
            cudaHostRegister(S_output_socket[i], 2 * (SelsNum + SelsNum/8) * sizeof(int), cudaHostRegisterMapped);
    }

    printf ("Setting up\n");

    output_host = (int*) numa_alloc_onnode(4*(SelsNum + SelsNum/8)*sizeof(int), numa_node_of_cpu(1));
    cudaHostRegister(output_host, 2*(SelsNum + SelsNum/8)*sizeof(int), cudaHostRegisterMapped);

    size_t* offsets_R_GPU;
    size_t* offsets_S_GPU[64];

    offsets_R_GPU = (size_t*) malloc(parts_host*4*OMP_PARALLELISM1*sizeof(size_t));

    for (int j = 0; j < S_segment_num; j++)
        offsets_S_GPU[j] = (size_t*) malloc(parts_host*4*OMP_PARALLELISM2*sizeof(size_t));

    printf ("Warmup\n");

    partition_prepare_payload (R, Pr, RelsNum, log_parts_host, first_bit + log_parts1 + log_parts2, 
                            R_socket, R_output_socket,
                            Pr_socket, Pr_output_socket,
                            R_offsets_socket, total_R, offsets_R_GPU, OMP_PARALLELISM1);

    for (int j = 0; j < S_segment_num; j++) {
        size_t size = (j == S_segment_num - 1) ? SelsNum - j * S_segment_size : S_segment_size;
        partition_prepare_payload (S + j * S_segment_size, Ps + j * S_segment_size, size, log_parts_host, first_bit + log_parts1 + log_parts2,
                            S_segment_socket[j], S_segment_output_socket[j],
                            Ps_segment_socket[j], Ps_segment_output_socket[j],
                            S_segment_offsets_socket[j], total_S[j], offsets_S_GPU[j], OMP_PARALLELISM2);
    }

    for (int i = 0; i < 2; i++) {
        memcpy (R_offsets_virt_socket[i], R_offsets_socket[i], (OMP_PARALLELISM1+1)*OMP_MEMORY_STEP*sizeof(size_t));
    }

    for (int i = 0; i < parts_host; i++)
            cudaMemcpy (offsets_R[i], offsets_R_GPU + i*4*OMP_PARALLELISM1, OMP_PARALLELISM1*4*sizeof(size_t), cudaMemcpyHostToDevice);

    for (int p = 0; p < PARTS_RESIDENT; p++)
        for (int t = 0; t < OMP_PARALLELISM1; t++)
            R_offsets_virt_socket[0][p  + t*OMP_MEMORY_STEP] += (R_output_socket[1] - R_output_socket[0]);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < S_segment_num; j++) {
            memcpy (S_segment_offsets_virt_socket[j][i], S_segment_offsets_socket[j][i], OMP_PARALLELISM2*OMP_MEMORY_STEP*sizeof(size_t));
        }
    }

    for (int i = 0; i < parts_host; i++)
        for (int j = 0; j < S_segment_num; j++)
            cudaMemcpy (offsets_S[i][j], offsets_S_GPU[j] + i*OMP_PARALLELISM2*4, OMP_PARALLELISM2*4*sizeof(size_t), cudaMemcpyHostToDevice);


    for (int j = 0; j < S_segment_num; j++)
        for (int p = 0; p < PARTS_RESIDENT; p++) {
            for (int t = 0; t < OMP_PARALLELISM2; t++)
                S_segment_offsets_virt_socket[j][0][p  + t*OMP_MEMORY_STEP] += (S_output_socket[1] - S_output_socket[0]);
        }

    printf ("Warmup\n");

    partition_do_payload (R_socket, R_output_socket, 
                            Pr_socket, Pr_output_socket,
                            R_offsets_virt_socket, RelsNum, log_parts_host, first_bit + log_parts1 + log_parts2, OMP_PARALLELISM1);

    for (int j = 0; j < S_segment_num; j++) {
        size_t size = (j == S_segment_num - 1) ? SelsNum - j * S_segment_size : S_segment_size;
        partition_do_payload (S_segment_socket[j], S_segment_output_socket[j], 
                                Ps_segment_socket[j], Ps_segment_output_socket[j], 
                                S_segment_offsets_virt_socket[j], size, log_parts_host, first_bit + log_parts1 + log_parts2, OMP_PARALLELISM2);
    }

    printf ("Begin join\n");

    cudaEvent_t prepare_R[16];
    cudaEvent_t done_R[16];
    cudaEvent_t done_S[16*64];
    cudaEvent_t prepare_S[16*64];
    cudaEvent_t init_R[16];
    cudaEvent_t output_S[16*64];

    for (int i = 0; i < parts_host; i++) {
        cudaEventCreate(&prepare_R[i]);
        cudaEventCreate(&init_R[i]);
        cudaEventCreate(&done_R[i]);

        for (int j = 0; j < S_segment_num; j++) {
            cudaEventCreate(&prepare_S[j*parts_host + i]);
            cudaEventCreate(&output_S[j*parts_host + i]);
            cudaEventCreate(&done_S[j*parts_host + i]);
        }
    }

    CHK_ERROR(cudaDeviceSynchronize());

    std::list<std::list<int> > schedule;
    /*
    //naive schedule
    {
        std::list<int> step1;
        step1.push_back(0); step1.push_back(1); step1.push_back(2); step1.push_back(3); step1.push_back(4);
        schedule.push_back(step1);
    }
    {
        std::list<int> step1;
        step1.push_back(5); step1.push_back(6); step1.push_back(7); step1.push_back(8); step1.push_back(9);
        schedule.push_back(step1);
    }
    {
        std::list<int> step1;
        step1.push_back(10); step1.push_back(11); step1.push_back(12); step1.push_back(13); step1.push_back(14);
        schedule.push_back(step1);
    }
    {
        std::list<int> step1;
        step1.push_back(15);
        schedule.push_back(step1);
    }*/

    double gain[16];
    for (int i = 0; i < parts_host; i++) {
        size_t len = (i < parts_host-1) ? R_offsets_socket[0][i+1] - R_offsets_socket[0][i] : total_R[0] - R_offsets_socket[0][i];
        gain[i] = ((double) len)/(buckets_num_max_R*bucket_size);
    }

    groupOptimal2 (gain, parts_host, schedule);

    int budget[PARTS_RESIDENT];
    int weight[16];
    int alias[PARTS_RESIDENT];
    int assign[PARTS_RESIDENT];

    for (int i = 0; i < parts_host; i++)
        weight[i] = ceil(gain[i]);


    printf ("Start...\n");

    uint32_t nstreams = 2;

    double t1 = cpuSeconds();

    partition_do_payload (R_socket, R_output_socket, 
                    Pr_socket, Pr_output_socket,
                    R_offsets_virt_socket, RelsNum, log_parts_host, first_bit + log_parts1 + log_parts2, OMP_PARALLELISM1);

    double t2 = cpuSeconds();

    uint32_t action_id = 0;
    uint32_t event_id = 0;

    int mincnt = RelsNum;
    int maxcnt = -1;
    int* hcnt = (int*) malloc(buckets_num_max_R*parts_host*S_segment_num*sizeof(int));

    size_t out_cnt = 0;

    for (int i = 0; i < PARTS_RESIDENT; i++)
        budget[i] = -1;

    bool init = true;

    for (std::list<std::list<int> >::iterator it = schedule.begin(); it != schedule.end(); ++it) {
        std::list<int>& batch = *it;

        int p = 0;
        int b = 0;

        for (std::list<int>::iterator it_inner = batch.begin(); it_inner != batch.end(); ++it_inner) {
            alias[p] = *it_inner;
            assign[p] = b;
            b += weight[*it_inner];
            p++;
        }

        std::list<int> large;

        for (int i = 0; i < p; i++) {
            int idx = alias[i];

            if (weight[idx] > 1 && i != p-1)
                large.push_back(idx);

            size_t len1 = R_offsets_socket[1][idx] - R_offsets_socket[0][idx];
            size_t len = (idx < parts_host-1) ? R_offsets_socket[0][idx+1] - R_offsets_socket[0][idx] : total_R[0] - R_offsets_socket[0][idx];
            size_t len2 = len - len1;

            uint64_t* heads[2];
            uint32_t* cnts[2];
            uint32_t* chains[2];
            uint32_t* buckets_used[2];

            heads[0] = heads_R[PARTS_RESIDENT+1-weight[idx]];
            cnts[0] = cnts_R[PARTS_RESIDENT+1-weight[idx]];
            chains[0] = chains_R[PARTS_RESIDENT+1-weight[idx]];
            buckets_used[0] = buckets_used_R[PARTS_RESIDENT+1-weight[idx]];    


            heads[1] = heads_R[assign[i]];
            cnts[1] = cnts_R[assign[i]];
            chains[1] = chains_R[assign[i]];
            buckets_used[1] = buckets_used_R[assign[i]];

            
            if (idx >= PARTS_RESIDENT) {
                size_t len1 = R_offsets_socket[1][idx] - R_offsets_socket[0][idx];

                numa_copy_multithread (
                    R_output_socket[1] + R_offsets_virt_socket[0][idx], 
                    R_output_socket[0] + R_offsets_virt_socket[0][idx], 
                    len1
                );

                numa_copy_multithread (
                    Pr_output_socket[1] + R_offsets_virt_socket[0][idx], 
                    Pr_output_socket[0] + R_offsets_virt_socket[0][idx], 
                    len1
                );


                for (int t = 0; t < OMP_PARALLELISM1; t++)
                    R_offsets_virt_socket[0][idx  + t*OMP_MEMORY_STEP] += (R_output_socket[1] - R_output_socket[0]);
            }
            

            for (int j = assign[i]; j < assign[i] + weight[idx]; j++) {
                if (budget[j] >= 0) {
                    cudaStreamWaitEvent (streams_R[0], done_R[budget[j]], 0);
                }
                budget[j] = idx;
            }

            if (i == p-1 && assign[p-1] + weight[alias[p-1]] >= PARTS_RESIDENT) {
                for (std::list<int>::iterator it = large.begin(); it != large.end(); ++it) {
                    cudaStreamWaitEvent (streams_R[0], init_R[*it], 0);
                }
            }

            
            cudaMemcpyAsync(R_gpu[assign[i]], R_output_socket[0] + R_offsets_virt_socket[0][idx],
                len * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

            cudaMemcpyAsync(Pr_gpu[assign[i]], Pr_output_socket[0] + R_offsets_virt_socket[0][idx],
                len * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

            cudaEventRecord (prepare_R[idx], streams_R[0]);

            cudaStreamWaitEvent (streams_R[1], prepare_R[idx], 0);
            
            prepare_Relation_payload (R_gpu[assign[i]], R_gpu[PARTS_RESIDENT+1-weight[idx]],
                            Pr_gpu[assign[i]], Pr_gpu[PARTS_RESIDENT+1-weight[idx]],
                            len,
                            buckets_num_max_R, heads, cnts, chains, buckets_used, 
                            log_parts1, log_parts2, 0, streams_R[1], offsets_R[idx], OMP_PARALLELISM1);

            cudaMemsetAsync(bucket_info_R[assign[i]], 0, buckets_num_max_R * sizeof(int), streams_R[1]);

            decompose_chains <<<1, 1024, 0, streams_R[1]>>> (bucket_info_R[assign[i]], chains_R[assign[i]], cnts_R[assign[i]], log_parts1 + log_parts2, 2*bucket_size);


            cudaEventRecord (init_R[idx], streams_R[1]);

            action_id++;
        }

        for (int j = 0; j < S_segment_num; j++) {
            if (init) {
                size_t size = (j == S_segment_num - 1) ? SelsNum - j * S_segment_size : S_segment_size;
                partition_do_payload (S_segment_socket[j], S_segment_output_socket[j], 
                                Ps_segment_socket[j], Ps_segment_output_socket[j], 
                                S_segment_offsets_virt_socket[j], size, log_parts_host, first_bit + log_parts1 + log_parts2, OMP_PARALLELISM2);
            }

            for (int i = 0; i < p; i++) {
                int idx = alias[i];

                size_t len1 = S_segment_offsets_socket[j][1][idx] - S_segment_offsets_socket[j][0][idx];
                
                size_t len = (idx < parts_host-1) ? 
                                S_segment_offsets_socket[j][0][idx+1] - S_segment_offsets_socket[j][0][idx] :
                                total_S[j][0] - S_segment_offsets_socket[j][0][idx];

                size_t len2 = len - len1;

                uint64_t* heads[2];
                uint32_t* cnts[2];
                uint32_t* chains[2];
                uint32_t* buckets_used[2];

                heads[0] = heads_S[2];
                cnts[0] = cnts_S[2];
                chains[0] = chains_S[2];
                buckets_used[0] = buckets_used_S[2];    

                heads[1] = heads_S[event_id % 2];
                cnts[1] = cnts_S[event_id % 2];
                chains[1] = chains_S[event_id % 2];
                buckets_used[1] = buckets_used_S[event_id % 2];

                
                if (idx >= PARTS_RESIDENT) {
                    size_t len1 = S_segment_offsets_socket[j][1][idx] - S_segment_offsets_socket[j][0][idx];

                    numa_copy_multithread (
                        S_segment_output_socket[j][1] + S_segment_offsets_virt_socket[j][0][idx], 
                        S_segment_output_socket[j][0] + S_segment_offsets_virt_socket[j][0][idx], 
                        len1
                    );

                    numa_copy_multithread (
                        Ps_segment_output_socket[j][1] + S_segment_offsets_virt_socket[j][0][idx], 
                        Ps_segment_output_socket[j][0] + S_segment_offsets_virt_socket[j][0][idx], 
                        len1
                    );

                    
                    for (int t = 0; t < OMP_PARALLELISM2; t++)
                        S_segment_offsets_virt_socket[j][0][idx + t*OMP_MEMORY_STEP] += (S_output_socket[1] - S_output_socket[0]); 
                }
                
                
                if (event_id >= 2)
                    cudaStreamWaitEvent (streams_R[0], done_S[event_id - 2], 0);
                
                cudaMemcpyAsync(S_gpu[event_id % 2], S_segment_output_socket[j][0] + S_segment_offsets_virt_socket[j][0][idx],
                    len * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

                cudaMemcpyAsync(Ps_gpu[event_id % 2], Ps_segment_output_socket[j][0] + S_segment_offsets_virt_socket[j][0][idx],
                    len * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

                cudaEventRecord (prepare_S[event_id], streams_R[0]);


                cudaStreamWaitEvent (streams_R[1], prepare_S[event_id], 0);
                
                prepare_Relation_payload (S_gpu[event_id % 2], S_gpu[2], 
                            Ps_gpu[event_id % 2], Ps_gpu[2],
                            len,
                            buckets_num_max_S, heads, cnts, chains, buckets_used, 
                            log_parts1, log_parts2, 0, streams_R[1], offsets_S[idx][j], OMP_PARALLELISM2);


                if (event_id >= 2) 
                    cudaStreamWaitEvent (streams_R[1], output_S[event_id-2], 0);

                join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu[assign[i]], Pr_gpu[assign[i]], chains_R[assign[i]], bucket_info_R[assign[i]],
                                    S_gpu[event_id % 2], Ps_gpu[event_id % 2], cnts_S[event_id % 2], chains_S[event_id % 2],
                                    log_parts1 + log_parts2, buckets_used_R[assign[i]], &aggr_cnt[idx*S_segment_num + j]);
                

                
                /*join_partitioned_results <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu[assign[i]], Pr_gpu[assign[i]], chains_R[assign[i]], bucket_info_R[assign[i]],
                                    S_gpu[event_id % 2], Ps_gpu[event_id % 2], cnts_S[event_id % 2], chains_S[event_id % 2],
                                    log_parts1 + log_parts2, buckets_used_R[assign[i]], &aggr_cnt[idx*S_segment_num + j], output[event_id % 2]);*/
                
                cudaEventRecord (done_S[event_id], streams_R[1]); 
                
                cudaStreamWaitEvent (streams_R[2], done_S[event_id], 0);

                //int out_size = 64000000; 

                //cudaMemcpyAsync(output_host + out_cnt, output[event_id % 2], out_size*sizeof(int), cudaMemcpyDeviceToHost, streams_R[2]); 

                //out_cnt += out_size;

                if (out_cnt > 4*SelsNum) {
                    out_cnt = 0;
                }

                cudaEventRecord (output_S[event_id], streams_R[2]);

                if (j == S_segment_num - 1)
                    cudaEventRecord (done_R[idx], streams_R[1]);

                action_id++;
                event_id++; 

                CHK_ERROR(cudaDeviceSynchronize());
            }
        }

        init = false; 
    }

    CHK_ERROR(cudaDeviceSynchronize());

    //cudaProfilerStop();

    std::cout << out_cnt << std::endl;

    double t3 = cpuSeconds();

    std::cout << "Total Throughput (Co-processing) " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;

    int32_t* results = (int32_t*) malloc(parts_host * S_segment_num *sizeof(int32_t));
    cudaMemcpy (results, aggr_cnt, parts_host * S_segment_num *sizeof(int32_t), cudaMemcpyDeviceToHost);

    CHK_ERROR(cudaDeviceSynchronize());

    int64_t results_sum = 0;
    int32_t maxres = -1;
    int32_t minres = 1 << 30;

    int mini = -1;
    int maxi = -1;

    int large_cnt = 0;

    for (int i = 0; i < parts_host * S_segment_num; i++) {
        results_sum += results[i];

        if (results[i] < minres) {
            minres = results[i];
            mini = i;
        }

        if (results[i] > maxres)
            maxres = results[i];

        

        
    }

    for (int i = 0; i < parts_host; i++) {
        maxcnt = 0;

        for (int j = 0; j < buckets_num_max_R; j++) {
            if (hcnt[j + buckets_num_max_R*i*S_segment_num] < mincnt) {
                mincnt = hcnt[j + buckets_num_max_R*i*S_segment_num];
            }

            if (hcnt[j + buckets_num_max_R*i*S_segment_num] > 0)
                maxcnt++;

            large_cnt += hcnt[j + buckets_num_max_R*i*S_segment_num] & ((1 << 15) - 1);
        }
    }

    printf ("%ld results\n", results_sum);
}

/*streaming-gpu case*/

unsigned int outOfGPU_Join3_payload (int* R, int* Pr, size_t RelsNum, int* S, int* Ps, size_t SelsNum, timingInfo *time, uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit) {
    cudaSetDevice(1);

    uint32_t parts1 = 1 << log_parts1;
    uint32_t parts2 = 1 << (log_parts1 + log_parts2);
    uint32_t parts_host = 1 << 4;

    cudaStream_t*  streams_R = (cudaStream_t*) malloc(3*sizeof(cudaStream_t));;

    cudaStreamCreate(&streams_R[0]);
    cudaStreamCreate(&streams_R[1]);
    cudaStreamCreate(&streams_R[2]);

    size_t S_segment_size = RelsNum / 4; 
    size_t S_segment_num = (SelsNum + S_segment_size -1)/S_segment_size;

    size_t buckets_num_max_R    = 2*((((RelsNum + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;// + parts2; 
    size_t buckets_num_max_S    = 2*((((S_segment_size + parts2 - 1)/parts2) + bucket_size - 1)/bucket_size)*parts2 + 1024;// + parts2;

    int32_t * aggr_cnt;
    CHK_ERROR(cudaMalloc((void**) &aggr_cnt,    1024 * parts_host * S_segment_num * sizeof(int32_t)));
    CHK_ERROR(cudaMemset(aggr_cnt, 0,           1024 * parts_host * S_segment_num * sizeof(int32_t)));

    
    int* R_gpu[2];
    int* S_gpu[3];

    int* Pr_gpu[2];
    int* Ps_gpu[3];
    
    uint32_t* chains_R[2];
    uint32_t* chains_S[3];

    uint32_t* cnts_R[2];
    uint32_t* cnts_S[3];

    uint64_t* heads_R[2];
    uint64_t* heads_S[3];

    uint32_t* buckets_used_R[2];
    uint32_t* buckets_used_S[3];

    size_t*   offsets_R;
    size_t*   offsets_S[64];

    int* output_host;

    int* R_buffer;
    int* Pr_buffer;
    uint32_t* chains_R_buffer;
    uint32_t* cnts_R_buffer;
    uint64_t* heads_R_buffer;
    uint32_t* buckets_used_R_buffer;
    size_t*   offsets_R_buffer;

    CHK_ERROR(cudaMalloc((void**) &R_buffer,                2 * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Pr_buffer,               2 * buckets_num_max_R * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &chains_R_buffer,         2 * buckets_num_max_R * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &cnts_R_buffer,           2 * parts2 * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &heads_R_buffer,          2 * parts2 * sizeof(uint64_t)));
    CHK_ERROR(cudaMalloc((void**) &buckets_used_R_buffer,   2 * sizeof(uint32_t)));

    CHK_ERROR(cudaMalloc((void**) &offsets_R_buffer,        OMP_PARALLELISM1 * 4 * sizeof(size_t)));

    for (int i = 0; i < 2; i++) {
        R_gpu[i]        = R_buffer + i * buckets_num_max_R * bucket_size;
        Pr_gpu[i]       = Pr_buffer + i * buckets_num_max_R * bucket_size;
        chains_R[i]     = chains_R_buffer + i * buckets_num_max_R;
        cnts_R[i]       = cnts_R_buffer + i * parts2;
        heads_R[i]      = heads_R_buffer + i * parts2;
        buckets_used_R[i] = buckets_used_R_buffer + i;
    }

    offsets_R = offsets_R_buffer;

    cudaMemset (R_buffer, 0, 2 * buckets_num_max_R * bucket_size * sizeof(int32_t));

    int* S_buffer;
    int* Ps_buffer;
    uint32_t* chains_S_buffer;
    uint32_t* cnts_S_buffer;
    uint64_t* heads_S_buffer;
    uint32_t* buckets_used_S_buffer;

    size_t* offsets_S_buffer;

    CHK_ERROR(cudaMalloc((void**) &S_buffer,                3 * buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &Ps_buffer,               3 * buckets_num_max_S * bucket_size * sizeof(int32_t)));
    CHK_ERROR(cudaMalloc((void**) &chains_S_buffer,         3 * buckets_num_max_S * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &cnts_S_buffer,           3 * parts2 * sizeof(uint32_t)));
    CHK_ERROR(cudaMalloc((void**) &heads_S_buffer,          3 * parts2 * sizeof(uint64_t)));
    CHK_ERROR(cudaMalloc((void**) &buckets_used_S_buffer,   3 * sizeof(uint32_t)));

    CHK_ERROR(cudaMalloc((void**) &offsets_S_buffer,        S_segment_num * OMP_PARALLELISM2 * 4 * sizeof(size_t)));

    for (int i = 0; i < 3; i++) {
        S_gpu[i]        = S_buffer + i * buckets_num_max_S * bucket_size;
        Ps_gpu[i]       = Ps_buffer + i * buckets_num_max_S * bucket_size;
        chains_S[i]     = chains_S_buffer + i * buckets_num_max_S;
        cnts_S[i]       = cnts_S_buffer + i * parts2;
        heads_S[i]      = heads_S_buffer + i * parts2;
        buckets_used_S[i] = buckets_used_S_buffer + i;
    }

    int32_t* output[2];

    output[0] = Pr_gpu[1];
    output[1] = Pr_gpu[1] + 2* S_segment_size;


    cudaMemset (S_buffer, 0, 3 * buckets_num_max_S * bucket_size * sizeof(int32_t));

    const uint64_t mask = 0x00000000000000FF;

    int* R_socket;
    int* Pr_socket;

    R_socket = (int*) numa_alloc_onnode((RelsNum+1024)*sizeof(int), numa_node_of_cpu(1));
    Pr_socket = (int*) numa_alloc_onnode((RelsNum+1024)*sizeof(int), numa_node_of_cpu(1));

    memcpy (R_socket, R, RelsNum *sizeof(int));
    memcpy (Pr_socket, Pr, RelsNum *sizeof(int));
 
    cudaHostRegister(R_socket, RelsNum * sizeof(int), cudaHostRegisterMapped);
    cudaHostRegister(Pr_socket, RelsNum * sizeof(int), cudaHostRegisterMapped);

    int* S_socket;
    int* Ps_socket;

    S_socket = (int*) numa_alloc_onnode((SelsNum + 4096)*sizeof(int), numa_node_of_cpu(1));
    Ps_socket = (int*) numa_alloc_onnode((SelsNum + 4096)*sizeof(int), numa_node_of_cpu(1));

    memcpy (S_socket, S, SelsNum *sizeof(int));
    memcpy (Ps_socket, Ps, SelsNum *sizeof(int));

    cudaHostRegister(S_socket, SelsNum * sizeof(int), cudaHostRegisterMapped);
    cudaHostRegister(Ps_socket, SelsNum * sizeof(int), cudaHostRegisterMapped);

    output_host = (int*) numa_alloc_onnode(2*SelsNum*sizeof(int), numa_node_of_cpu(1));
    cudaHostRegister(output_host, 2*SelsNum*sizeof(int), cudaHostRegisterMapped);

    printf ("Begin join\n");

    cudaEvent_t prepare_R[1];
    cudaEvent_t done_R[1];
    cudaEvent_t done_S[64];
    cudaEvent_t prepare_S[64];
    cudaEvent_t init_R[1];
    cudaEvent_t output_S[64];

    cudaEventCreate(&prepare_R[0]);
    cudaEventCreate(&init_R[0]);
    cudaEventCreate(&done_R[0]);

    for (int j = 0; j < S_segment_num; j++) {
        cudaEventCreate(&prepare_S[j]);
        cudaEventCreate(&output_S[j]);
        cudaEventCreate(&done_S[j]);
    }


    CHK_ERROR(cudaDeviceSynchronize());

    uint32_t nstreams = 2;

    double t1 = cpuSeconds();

    uint32_t action_id = 0;
    uint32_t event_id = 0;

    size_t out_cnt = 0;

    uint32_t* bucket_info_R = (uint32_t*) R_gpu[1];

    {
        uint64_t* heads[2];
        uint32_t* cnts[2];
        uint32_t* chains[2];
        uint32_t* buckets_used[2];

        heads[0] = heads_R[1];
        cnts[0] = cnts_R[1];
        chains[0] = chains_R[1];
        buckets_used[0] = buckets_used_R[1];    

        heads[1] = heads_R[0];
        cnts[1] = cnts_R[0];
        chains[1] = chains_R[0];
        buckets_used[1] = buckets_used_R[0];

        cudaMemcpyAsync(R_gpu[0], R_socket,
            RelsNum * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

        cudaMemcpyAsync(Pr_gpu[0], Pr_socket,
            RelsNum * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

        cudaEventRecord (prepare_R[0], streams_R[0]);

        cudaStreamWaitEvent (streams_R[1], prepare_R[0], 0);
                
        prepare_Relation_payload (R_gpu[0], R_gpu[1],
                                Pr_gpu[0], Pr_gpu[1],
                                RelsNum,
                                buckets_num_max_R, heads, cnts, chains, buckets_used, 
                                log_parts1, log_parts2, 0, streams_R[1], NULL, OMP_PARALLELISM1);

        decompose_chains <<<1, 1024, 0, streams_R[1]>>> (bucket_info_R, chains_R[0], cnts_R[0], log_parts1 + log_parts2, 2*bucket_size); 

        cudaEventRecord (init_R[0], streams_R[1]);

        action_id++;
    }

    for (int j = 0; j < S_segment_num; j++) {
        size_t size = (j == S_segment_num - 1) ? SelsNum - j * S_segment_size : S_segment_size;
    
        uint64_t* heads[2];
        uint32_t* cnts[2];
        uint32_t* chains[2];
        uint32_t* buckets_used[2];

        heads[0] = heads_S[2];
        cnts[0] = cnts_S[2];
        chains[0] = chains_S[2];
        buckets_used[0] = buckets_used_S[2];    

        heads[1] = heads_S[event_id % 2];
        cnts[1] = cnts_S[event_id % 2];
        chains[1] = chains_S[event_id % 2];
        buckets_used[1] = buckets_used_S[event_id % 2];



        if (event_id >= 2)
            cudaStreamWaitEvent (streams_R[0], done_S[event_id - 2], 0);

        cudaMemcpyAsync(S_gpu[event_id % 2], S_socket + j * S_segment_size,
            size * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

        cudaMemcpyAsync(Ps_gpu[event_id % 2], Ps_socket + j * S_segment_size,
            size * sizeof(int), cudaMemcpyHostToDevice, streams_R[0]);

        cudaEventRecord (prepare_S[event_id], streams_R[0]);


        cudaStreamWaitEvent (streams_R[1], prepare_S[event_id], 0);
                
        prepare_Relation_payload (S_gpu[event_id % 2], S_gpu[2], 
                            Ps_gpu[event_id % 2], Ps_gpu[2],
                            size,
                            buckets_num_max_S, heads, cnts, chains, buckets_used, 
                            log_parts1, log_parts2, 0, streams_R[1], NULL, OMP_PARALLELISM2);


        if (event_id >= 2)
            cudaStreamWaitEvent (streams_R[1], output_S[event_id-2], 0);


        /*join_partitioned_aggregate <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu[0], Pr_gpu[0], chains_R[0], bucket_info_R,
                                    S_gpu[event_id % 2], Ps_gpu[event_id % 2], cnts_S[event_id % 2], chains_S[event_id % 2],
                                    log_parts1 + log_parts2, buckets_used_R[0], &aggr_cnt[j]);*/

        join_partitioned_results <<<(1 << log_parts1), 512, 0, streams_R[1]>>> (
                                    R_gpu[0], Pr_gpu[0], chains_R[0], bucket_info_R,
                                    S_gpu[event_id % 2], Ps_gpu[event_id % 2], cnts_S[event_id % 2], chains_S[event_id % 2],
                                    log_parts1 + log_parts2, buckets_used_R[0], &aggr_cnt[j], output[event_id % 2]);

        cudaEventRecord (done_S[event_id], streams_R[1]);
                
        cudaStreamWaitEvent (streams_R[2], done_S[event_id], 0);

        cudaMemcpyAsync(output_host + out_cnt, output[event_id % 2], 2*size*sizeof(int), 
                            cudaMemcpyDeviceToHost, streams_R[2]);

        out_cnt += 2*size;

        cudaEventRecord (output_S[event_id], streams_R[2]);

        action_id++;
        event_id++;
    }


    CHK_ERROR(cudaDeviceSynchronize());

    double t3 = cpuSeconds();

    std::cout << "Total Throughput (Streaming) " << (2 * (RelsNum + SelsNum)*sizeof(int))/(t3-t1)/1000/1000 << std::endl;

    int32_t* results = (int32_t*) malloc(parts_host * S_segment_num *sizeof(int32_t));
    cudaMemcpy (results, aggr_cnt, parts_host * S_segment_num *sizeof(int32_t), cudaMemcpyDeviceToHost);

    CHK_ERROR(cudaDeviceSynchronize());

    int64_t results_sum = 0;
    for (int i = 0; i < S_segment_num; i++)
        results_sum += results[i];

    printf ("%ld results\n", results_sum);
}





unsigned int hj_ClusteredProbe(int *R, size_t RelsNum, int *S, size_t SelsNum, timingInfo *time){
	int* Pr = (int*) malloc (RelsNum * sizeof(int));
    int* Ps = (int*) malloc (SelsNum * sizeof(int));
    
    for (size_t i = 0; i < RelsNum; i++) {
        Pr[i] = 1;
    }

    for (size_t i = 0; i < SelsNum; i++)
        Ps[i] = 1;
    
    if (RelsNum < 128000001) {
        if (SelsNum < 128000001) {
            outOfGPU_Join1_payload (R, Pr, RelsNum, S, Ps, SelsNum, NULL, log_parts1, log_parts2, 5 + p_d);
        } else {
            outOfGPU_Join3_payload (R, Pr, RelsNum, S, Ps, SelsNum, NULL, log_parts1, log_parts2, 5 + p_d);
        }        
    } else {
        outOfGPU_Join2_payload (R, Pr, RelsNum, S, Ps, SelsNum, NULL, log_parts1, log_parts2, 5 + p_d); 
    }
    return 0;
}

constexpr int log_htsize = 20;
int32_t ht[1 << log_htsize];

uint32_t h_hashMurmur(uint32_t x){
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x & ((1 << log_htsize) - 1);
}

void joinCpu(int32_t * __restrict__ R, size_t R_N, int32_t * __restrict__ S, size_t S_N){
    memset(ht, -1, sizeof(int32_t) * (1 << log_htsize));

    int32_t * next = (int32_t *) malloc(R_N * sizeof(int32_t));

    uint32_t g = 0;
    uint32_t s = 0;
    uint32_t c = 0;
    {
        time_block timer("TcpuJoin: ");
        for (size_t j = 0 ; j < R_N ; ++j){
            uint32_t bucket = h_hashMurmur(R[j]);
            next[j]    = ht[bucket];
            ht[bucket] = j;
            ++c;
        }
        std::cout << "===" << c << std::endl;

        #pragma omp parallel for reduction(+:g) reduction(+:c)
        for (size_t j = 0 ; j < S_N ; ++j){
            uint32_t bucket  = h_hashMurmur(S[j]);
            int32_t  current = ht[bucket];
            while (current >= 0){
                if (S[j] == R[current]) {
                    g += S[j];
                    ++s;
                }
                current = next[current];
            }
            ++c;
        }
    }
    std::cout << "===" << s << " " << c << " " << g << std::endl;
    std::cout << g << " join results" << std::endl;
}


unsigned int hashJoinClusteredProbe(args *inputAttrs, timingInfo *time) {
    fflush(stdout); //force printf

    uint64_t totalJoinsNum = hj_ClusteredProbe(inputAttrs->R, inputAttrs->R_els, inputAttrs->S, inputAttrs->S_els, time); //FIXME: check order of R/S

    fflush(stdout);

    recordTime(&(time->start[time->n-2]));
    recordTime(&(time->end[time->n-2]));

    return totalJoinsNum;
}
