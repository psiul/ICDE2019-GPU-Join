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

#ifndef PARTITION_PRIMITIVES_HPP_
#define PARTITION_PRIMITIVES_HPP_

#include <cassert>
#include <iostream>
#include <numa.h>
#include <unistd.h>

#include <list>

#include <omp.h>
#include <immintrin.h>

#include "common.h"
#include "common-host.h"

#define OMP_PARALLELISM1 16
#define OMP_PARALLELISM2 16 
#define OMP_MEMORY_STEP 4096
#define LOG_PARTS_OUTER 4
#define PARTS_RESIDENT 5

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
                    const uint32_t           nthreads);

void partition_prepare_payload (int* R, int* P, size_t n, uint32_t log_parts, uint32_t first_bit, 
                            int* R_sock[2], int* out_sock[2],
                            int* P_sock[2], int* pout_sock[2],
                            size_t* out_offsets[2], size_t total[2], size_t* offsets_GPU, uint32_t num_threads);

void partition_do_payload (int* R_sock[2], int* out_sock[2], int* P_sock[2], int* pout_sock[2], size_t* out_offsets[2], size_t n, uint32_t log_parts, uint32_t first_bit, uint32_t num_threads);

void numa_copy_multithread (int* __restrict__ dest, int* __restrict__ src, int n);

void sort (int* key, int* val, int n);

void shuffle (std::list<int>& chosen, int* weight_global, int maxw, std::list<int>& output);

void knapSack (std::list<int>& candidates, int* weight_global, double* gain_global, std::list<int>& output, std::list<int>& remainder);

void groupOptimal (double* gain, int n, std::list<std::list<int> >& output);

void groupOptimal2 (double* gain, int n, std::list<std::list<int> >& output);

#endif