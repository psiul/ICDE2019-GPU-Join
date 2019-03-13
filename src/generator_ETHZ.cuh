/*
Code adapted from  multicore-hashjoins-0.2@https://www.systems.ethz.ch/node/334
All credit to the original author: Cagri Balkesen <cagri.balkesen@inf.ethz.ch>
*/

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
