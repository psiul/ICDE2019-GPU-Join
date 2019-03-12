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

#include "partition-primitives.cuh"

#define LOG_BATCH 8
#define PARTS_CAPACITY 16

/*CPU-side partitioning
we assume that we already have histograms for M-way partitioning, where M is small like 16 (DBs keep statistics anyway so we do it in single pass)

S=keys of relation S
P=payload of relation S
out_cnts=count of elements for each partition
output_S=that's where we write partitioned keys
output_P=that's where we write partitioned payloads
cnt=total number of elements in relation
log_parts=number of partitions, logarithmic
first_bit=we shift right before taking bits for radix partitioning
nthreads=the number of threads running this, it helps find out what each thread reads
*/
void partitions_host_omp_nontemporal_payload(
                    const int32_t   * __restrict__ S,
                    const int32_t   * __restrict__ P,
                    size_t  * __restrict__ out_cnts,
                    int32_t   * __restrict__ output_S,
                    int32_t   * __restrict__ output_P,
                    const size_t             cnt,
                    const uint32_t           log_parts,
                    const uint32_t           first_bit,
                    const uint32_t           threadIdx,
                    const uint32_t           nthreads) {
    const uint32_t parts     = 1 << log_parts;
    const int32_t parts_mask = parts - 1;
    const int32_t bucket_mask = (1 << log2_bucket_size) - 1;

    size_t out_cnts_local[PARTS_CAPACITY];
    for (int i = 0; i < parts; i++) {
        out_cnts_local[i] = out_cnts[threadIdx*OMP_MEMORY_STEP + i];
    }

    /*software-managed caches, they have to be aligned for AVX2*/
    int32_t* cache_S = (int32_t*) aligned_alloc(4096, parts*(1 << LOG_BATCH)*sizeof(int32_t));
    int32_t* cache_P = (int32_t*) aligned_alloc(4096, parts*(1 << LOG_BATCH)*sizeof(int32_t));

    uint32_t regptr[PARTS_CAPACITY];
    for (int i = 0; i < parts; i++)
        regptr[i] = i << LOG_BATCH;
    
    for (size_t t = threadIdx; t < (cnt + OMP_MEMORY_STEP - 1)/OMP_MEMORY_STEP; t += nthreads) {
        const int32_t* chunk_S =  S + t*OMP_MEMORY_STEP;
        const int32_t* chunk_P =  P + t*OMP_MEMORY_STEP;

        int end = ((t+1)*OMP_MEMORY_STEP < cnt) ?
                        OMP_MEMORY_STEP :
                        cnt - t*OMP_MEMORY_STEP;
        //#pragma loop unroll
        for (int i = 0; i < end; i++) {
            int32_t key = chunk_S[i];
            int32_t payload = chunk_P[i];
            uint32_t partition =  (hasht(key) >> first_bit) & parts_mask;

            /*write element to cache*/
            uint32_t offset = (regptr[partition])++;
            cache_S[offset] = key;
            cache_P[offset] = payload;

            /*cache for partition is full, flush it to memory
            do it with non-temporal writes in order to avoid reading the output locations first*/
            if ((offset & ((1 << LOG_BATCH) - 1)) == ((1 << LOG_BATCH) - 1)) {
                for (int k = 0; k < (1 << (LOG_BATCH - 3)); k++) {
                    __m256i data = *((__m256i*) &cache_S[(partition << LOG_BATCH) + k*8]);
                    _mm256_stream_si256 ((__m256i*) &output_S[out_cnts_local[partition] + k*8], data);
                }

                for (int k = 0; k < (1 << (LOG_BATCH - 3)); k++) {
                    __m256i data = *((__m256i*) &cache_P[(partition << LOG_BATCH) + k*8]);
                    _mm256_stream_si256 ((__m256i*) &output_P[out_cnts_local[partition] + k*8], data);
                }

                out_cnts_local[partition] += (1 << LOG_BATCH);
                regptr[partition] = partition << LOG_BATCH;
            }
        }
    }

    /*flush half-full caches*/
    for (int p = 0; p < parts; p++) {
        for (int k = 0; k < (1 << (LOG_BATCH - 3)); k++) {
            if (8*k < regptr[p] - (p << LOG_BATCH)) {
              __m256i data = *((__m256i*) &cache_S[(p << LOG_BATCH) + k*8]);
                _mm256_stream_si256 ((__m256i*) &output_S[out_cnts_local[p] + k*8], data);
            }
        }

        for (int k = 0; k < (1 << (LOG_BATCH - 3)); k++) {
            if (8*k < regptr[p] - (p << LOG_BATCH)) {
              __m256i data = *((__m256i*) &cache_P[(p << LOG_BATCH) + k*8]);
               _mm256_stream_si256 ((__m256i*) &output_P[out_cnts_local[p] + k*8], data);
            }
        }

        out_cnts_local[p] += regptr[p] - (p << LOG_BATCH);
    }

    #pragma omp barrier
}

/*compute the offsets at which each thread writes by doing the count and then doing prefix sum
this is not part of runtime measurements*/
void partition_prepare_payload (int* R, int* P, size_t n, uint32_t log_parts, uint32_t first_bit, 
                            int* R_sock[2], int* out_sock[2],
                            int* P_sock[2], int* pout_sock[2],
                            size_t* out_offsets[2], size_t total[2], size_t* offsets_GPU, uint32_t num_threads) {
    uint32_t parts = (1 << log_parts);
    uint32_t parts_mask = parts - 1;

    #pragma omp parallel num_threads(num_threads) 
    {
        uint32_t threadIdx = omp_get_thread_num();
        uint32_t socket = sched_getcpu() % 2;

        for (size_t t = threadIdx; t < (n + OMP_MEMORY_STEP - 1)/OMP_MEMORY_STEP; t += num_threads) {
            int end = ((t+1)*OMP_MEMORY_STEP < n) ?
                        OMP_MEMORY_STEP :
                        n - t*OMP_MEMORY_STEP;

            for (int i = 0; i < end; i++) {
                R_sock[socket][t*OMP_MEMORY_STEP + i] = R[t*OMP_MEMORY_STEP + i];
                P_sock[socket][t*OMP_MEMORY_STEP + i] = P[t*OMP_MEMORY_STEP + i];

                uint32_t partition =  (hasht(R[t*OMP_MEMORY_STEP + i]) >> first_bit) & parts_mask;
                out_offsets[socket][partition + threadIdx*OMP_MEMORY_STEP] += 1;
            }
        }
    }

    size_t prefix1 = 0;

    for (int i = 0; i < parts; i++) {
        size_t base = prefix1;

        for (int j = 0; j < num_threads; j++) {
            size_t temp = out_offsets[0][i + j*OMP_MEMORY_STEP];
            out_offsets[0][i + j*OMP_MEMORY_STEP] = prefix1;

            offsets_GPU[i*num_threads*4 + 2*j] = prefix1 - base;

            prefix1 += temp;

            offsets_GPU[i*num_threads*4 + 2*j + 1] = prefix1 - base;

            prefix1 = ((prefix1 + 31)/32)*32;
        }

        for (int j = 0; j < num_threads; j++) {
            size_t temp = out_offsets[1][i + j*OMP_MEMORY_STEP];
            out_offsets[1][i + j*OMP_MEMORY_STEP] = prefix1;

            offsets_GPU[i*num_threads*4 + num_threads*2 + 2*j] = prefix1 - base;

            prefix1 += temp;

            offsets_GPU[i*num_threads*4 + num_threads*2 + 2*j + 1] = prefix1 - base;
            

            prefix1 = ((prefix1 + 31)/32)*32;
        }

        double fraction = ((double) (prefix1 - base))/n;
    }

    total[0] = prefix1;
    total[1] = prefix1;

    #pragma omp parallel num_threads(num_threads)
    {
        uint32_t threadIdx = omp_get_thread_num();
        uint32_t socket = sched_getcpu() % 2;

        /*test run, I use it to warm up the memory (make sure it is allocated by the time I access it)*/
        partitions_host_omp_nontemporal_payload(R_sock[socket], P_sock[socket], out_offsets[socket], out_sock[socket], pout_sock[socket], n, log_parts, first_bit, threadIdx, num_threads);
        #pragma omp barrier
    }


    double t1 = cpuSeconds();

    #pragma omp parallel num_threads(num_threads)
    {
        uint32_t threadIdx = omp_get_thread_num();
        uint32_t socket = sched_getcpu() % 2;

        partitions_host_omp_nontemporal_payload(R_sock[socket], P_sock[socket], out_offsets[socket], out_sock[socket], pout_sock[socket], n, log_parts, first_bit, threadIdx, num_threads);
        #pragma omp barrier
    }

    double t2 = cpuSeconds();

    printf("bw %f MB/s\n", (n * sizeof(int)) / 1000000 / (t2 - t1));

}

/*this function handles the multithreaded partitioning*/
void partition_do_payload (int* R_sock[2], int* out_sock[2], int* P_sock[2], int* pout_sock[2], size_t* out_offsets[2], size_t n, uint32_t log_parts, uint32_t first_bit, uint32_t num_threads) {
    #pragma omp parallel num_threads(num_threads)
    {
        uint32_t threadIdx = omp_get_thread_num();
        uint32_t socket = sched_getcpu() % 2;

        partitions_host_omp_nontemporal_payload(R_sock[socket], P_sock[socket], out_offsets[socket], out_sock[socket], pout_sock[socket], n, log_parts, first_bit, threadIdx, num_threads);
        #pragma omp barrier
    }
}

/*this function handles the multithreaded numa copy (useful for staging between sockets before transfer). I use only some thread to avoid eating away bandwidth from PCIe*/
void numa_copy_multithread (int* __restrict__ dest, int* __restrict__ src, int n) {
    #pragma omp parallel num_threads(OMP_PARALLELISM2) 
    {
        uint32_t threadIdx = omp_get_thread_num() % (OMP_PARALLELISM2/2);
        uint32_t socket = sched_getcpu() % 2;

        if (socket == 1)
            for (size_t t = threadIdx; t < (n + OMP_MEMORY_STEP - 1)/OMP_MEMORY_STEP; t += OMP_PARALLELISM2/2) {
                int end = ((t+1)*OMP_MEMORY_STEP < n) ?
                            OMP_MEMORY_STEP :
                            n - t*OMP_MEMORY_STEP;

                for (int i = 0; i < end; i += 8) {
                    __m256i data = _mm256_load_si256 ((__m256i*) &src[t*OMP_MEMORY_STEP+i]);
                    _mm256_stream_si256 ((__m256i*) &dest[t*OMP_MEMORY_STEP+i], data);
                }
            }
    }
}


/*functions used to find which partitions to batch together*/


void sort (int* key, int* val, int n) {
    if (n <= 1)
        return;

    int k = 1;
    int pivot = key[0];

    for (int i = 1; i < n; i++) {
        if (key[i] >= pivot) {
            int temp = key[i];
            key[i] = key[k];
            key[k] = temp;
            k++;
        }
    }

    key[0] = key[k-1];
    key[k-1] = pivot;

    sort (key, val, k-1);
    sort (key + k, val + k, n-k);
}

void shuffle (std::list<int>& chosen, int* weight_global, int maxw, std::list<int>& output) {
    int n = chosen.size();
    int cnt = 0;
    int totalw = 0;

    int* alias = new int[n];
    int* weight = new int[n];

    for (std::list<int>::iterator it = chosen.begin(); it != chosen.end(); ++it) {
        alias[cnt] = *it;
        weight[cnt] = weight_global[*it];
        totalw = totalw + weight[cnt];
        cnt++;
    }

    sort (weight, alias, n);

    for (int i = 0; i < n; i++)
        output.push_back(alias[i]);

    delete[] alias;
    delete[] weight;
}


void knapSack (std::list<int>& candidates, int* weight_global, double* gain_global, std::list<int>& output, std::list<int>& remainder) {
    int n = candidates.size();
    int w = PARTS_RESIDENT+1;
    int cnt = 0;

    int* weight = new int[n];
    double* gain = new double[n];
    int* alias = new int[n];

    for (std::list<int>::iterator it = candidates.begin(); it != candidates.end(); ++it) {
        alias[cnt] = *it;
        gain[cnt] = gain_global[*it];
        weight[cnt] = weight_global[*it];
        cnt++;
    }


    double** matrix = new double*[n+1];

    for (int i = 0; i < n+1; i++) {
        matrix[i] = new double[w+1];

        for (int j = 0; j < w+1; j++)
            matrix[i][j] = 0.0;
    }

    for (int i = 0; i < n+1; i++) {
        int wt = (i > 0)? weight[i-1] : 0;
        double g = (i > 0)? gain[i-1] : 0.0;

        for (int j = 0; j < w+1; j++) {
            if (i == 0 || j == 0)
                matrix[i][j] = 0.0;
            else if (wt <= j)
                matrix[i][j] = (matrix[i-1][j] + 0.000001 < matrix[i-1][j-wt] + g)? matrix[i-1][j-wt] + g : matrix[i-1][j];
            else
                matrix[i][j] = matrix[i-1][j];
        }
    }

    int t = PARTS_RESIDENT;
    int m = n;
    std::list<int> pr_output;

    while (t > 0 && m != 0) {
        for (int i = m; i > 0; i--)
            if (matrix[i][t] > matrix[i-1][t] + 0.000001) {
                pr_output.push_back(alias[i-1]);
                t -= weight[i-1];
                m = i-1;
                break;
            } else {
                remainder.push_back(alias[i-1]);
            }
    }

    shuffle (pr_output, weight_global, PARTS_RESIDENT, output);

    for (int i = m; i > 0; i--) {
        remainder.push_back(alias[i-1]);
    }

    delete[] weight;
    delete[] gain;
    delete[] alias;

    for (int i = 0; i < n+1; i++)
        delete[] matrix[i];

    delete[] matrix;
}

#include <cmath>

void groupOptimal (double* gain, int n, std::list<std::list<int> >& output) {
    std::list<int> candidates;
    int* weight = new int[n];

    for (int i = 0; i < n; i++) {
        candidates.push_back(i);
        weight[i] = ceil(gain[i]);
    }

    while (candidates.empty() == false) {
        std::list<int> out;
        std::list<int> remainder;

        knapSack (candidates, weight, gain, out, remainder);

        output.push_back(out);

        candidates = remainder;
    }


    delete[] weight;
}

void groupOptimal2 (double* gain, int n, std::list<std::list<int> >& output) {
    std::list<int> candidates;
    std::list<int>* buckets = new std::list<int> [2];
    int* weight = new int[n];

    for (int i = 0; i < n; i++) {
        candidates.push_back(i);
        weight[i] = ceil(gain[i]);
    }

    std::list<int> out;
    std::list<int> remainder;
    knapSack (candidates, weight, gain, out, remainder);

    output.push_back(out);

    for (std::list<int>::iterator it = remainder.begin(); it != remainder.end(); ++it) {
        buckets[weight[*it] - 1].push_back(*it);
    }

    std::list<std::list<int> > out2;
    for (std::list<int>::iterator it = buckets[1].begin(); it != buckets[1].end(); ++it) {
        std::list<int> new_out;
        new_out.push_back(*it);
        out2.push_back(new_out);
    }

    for (std::list<std::list<int> >::iterator it = out2.begin(); it != out2.end(); ++it) {
        for (int i = 0; i < 3; i++) {
            int next = buckets[0].front();
            (*it).push_back(next);
            buckets[0].pop_front();

            if (buckets[0].empty())
                break;
        }

        output.push_back(*it);

        if (buckets[0].empty()) {
            ++it;

            while (it != out2.end()) {
                output.push_back(*it);
                ++it;
            }
            
            break;
        }
    }

    while (buckets[0].empty() == false) {
        std::list<int> last;

        while (last.size() < PARTS_RESIDENT && buckets[0].empty() == false) {
            int next = buckets[0].front();
            last.push_back(next);
            buckets[0].pop_front();
        }

        output.push_back(last);
    }

    delete[] weight;
}