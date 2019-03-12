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

#ifndef GENERATOR_ETHZ_CUH_
#define GENERATOR_ETHZ_CUH_

#include <cstdint>	/*uint64_t*/

void seed_generator(unsigned int seed);

int readFromFile(const char * filename, int *relation, uint64_t num_tuples);
int create_relation_nonunique(const char *filename, int *relation, uint64_t num_tuples, const int64_t maxid);
int create_relation_unique(const char *filename, int *relation, uint64_t num_tuples, const int64_t maxid);
void random_gen(int *rel, uint64_t elsNum, const int64_t maxid);
void random_unique_gen(int *rel, uint64_t elsNum, const int64_t maxid);
int create_relation_fk_from_pk(const char *filename, int *fkrel, uint64_t fkrelElsNum, int *pkrel, uint64_t pkrelElsNum);
void knuth_shuffle(int *relation, uint64_t elsNum);
void knuth_shuffle48(int *relation, uint64_t elsNum, unsigned short * state);
int create_relation_zipf(const char *filename, int *relation, uint64_t elsNum, const int64_t maxid, const double zipf_param);
void gen_zipf(uint64_t stream_size, unsigned int alphabet_size, double zipf_factor, int *ret);
int create_relation_n(int* in_relation, int* out_relation, uint64_t num_tuples, uint64_t n);

#endif /* GENERATOR_ETHZ-CUH_ */
