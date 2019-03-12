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

#ifndef JOIN_PRIMITIVES_HPP_
#define JOIN_PRIMITIVES_HPP_

#include <cinttypes>
#include "common.h"
#include "common-host.h"

#define CLUSTERING_FACTOR 64

struct alignas(alignof(int64_t)) hj_bucket_2{
    int32_t next;
    int32_t val ;

    constexpr __host__ __device__ hj_bucket_2(int32_t next, int32_t value): next(next), val(value){}
};

__global__ void init_payload (int* R, int n);

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
                                          uint32_t                 num_threads);

__global__ void compute_bucket_info (uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts);

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
                                          uint32_t  *              bucket_num_ptr);

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
                                    int32_t*                      results);

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
                                    int32_t*                     buffer);

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
                                );

__global__ void build_perfect_array (int32_t* data, int32_t* payload, int n, int32_t* lookup);

__global__ void probe_perfect_array (int32_t* data, int32_t* payload, int n, int32_t* lookup, int* aggr);

__global__ void build_ht_chains (int32_t* data, int n, uint32_t log_parts, int32_t* output, int* head);

__global__ void chains_probing (int32_t* data, int32_t* payload, int n, uint32_t log_parts, int32_t* ht, int32_t* ht_key, int32_t* ht_pay, int* head, int* aggr);

__global__ void ht_hist (int* data, int n, int log_parts, int* hist);

__global__ void ht_offsets (int log_parts, int* hist, int* offset, int* aggr);

__global__ void build_ht_linear (int* data, int* payload, size_t n, int log_parts, int* offset, int* ht, int* htp);

__global__ void linear_probing (int* data, int* payload, int* ht, int* htp, int* offset_s, int* offset_e, size_t n, int log_parts, int* aggr);

__global__ void decompose_chains (uint32_t* bucket_info, uint32_t* chains, uint32_t* out_cnts, uint32_t log_parts, int threshold);

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
                                    int32_t*                     results);

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
                                    int32_t*                     output);


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
                                    int32_t*                     results);

__global__ void probe_perfect_array_varpay (int32_t* data, int32_t* Dr, int n, int32_t* lookup, int32_t* Ds, int col_num1, int col_num2, int res_size, int* aggr);

void prepare_Relation_payload (int* R, int* R_temp, int* P, int* P_temp, size_t RelsNum, uint32_t buckets_num, uint64_t* heads[2], uint32_t* cnts[2], uint32_t* chains[2], uint32_t* buckets_used[2], uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit, cudaStream_t streams, size_t* offsets_GPU, uint32_t num_threads);

void prepare_Relation_payload_triple (int* R, int* R_temp, int* R_final, int* P, int* P_temp, int* P_final, size_t RelsNum, uint32_t buckets_num, uint64_t* heads[2], uint32_t* cnts[2], uint32_t* chains[2], uint32_t* buckets_used[2], uint32_t log_parts1, uint32_t log_parts2, uint32_t first_bit, cudaStream_t streams, size_t* offsets_GPU, uint32_t num_threads);




#endif