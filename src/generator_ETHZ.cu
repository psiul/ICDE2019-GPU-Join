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

#include "generator_ETHZ.cuh"

#include <cstdint>
#include <cassert>
#include <cstdio>	/*printf*/
#include <cstdlib>
#include <ctime>
#include <cstring>
#include <cmath>

#define RAND_RANGE(N) ((double)rand() / ((double)RAND_MAX + 1) * (N))
#define RAND_RANGE48(N,STATE) ((double)nrand48(STATE)/((double)RAND_MAX+1)*(N))


static int seeded = 0;
static unsigned int seedValue;

void seed_generator(unsigned int seed) {
	srand(seed);
	seedValue = seed;
	seeded = 1;
}

/** Check whether seeded, if not seed the generator with current time */
static void check_seed() {
	if (!seeded) {
		seedValue = time(NULL);
		srand(seedValue);
		seeded = 1;
	}
}

int readFromFile(const char * filename, int *relation, uint64_t num_tuples) {
	char path[100];
	sprintf(path, "%s", filename);
	FILE *fp = fopen(path, "rb");

	if (!fp) return 1;

	printf("Reading file %s ", path);
	fflush(stdout);

	fread(relation, sizeof(int), num_tuples, fp);

	/*for (int i = 0; i < num_tuples; i++) {
        int k = rand() % num_tuples;
        int tmp = relation[k];
        relation[k] = relation[i];
        relation[i] = tmp;
    }*/

	fclose(fp);
	return 0;
}

static int writeToFile(const char * filename, int *relation, uint64_t num_tuples) {
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;

	fwrite(relation, sizeof(int), num_tuples, fp);
	fclose(fp);

	char path[100];
	sprintf(path, "%s", filename);
	rename(filename, path);
	return 0;
}

int create_relation_nonunique(const char *filename, int *relation, uint64_t num_tuples, const int64_t maxid) {
	/*first try to read from a file*/
	if (readFromFile(filename, relation, num_tuples)) {
		check_seed();
		random_gen(relation, num_tuples, maxid);

		return writeToFile(filename, relation, num_tuples);
	}
	return 0;
}


int create_relation_unique(const char *filename, int *relation, uint64_t num_tuples, const int64_t maxid) {
	/*first try to read from a file*/
	if (readFromFile(filename, relation, num_tuples)) {
		random_unique_gen(relation, num_tuples, maxid);
		return writeToFile(filename, relation, num_tuples);
	}

	return 0;
}


int create_relation_n(int* in_relation, int* out_relation, uint64_t num_tuples, uint64_t n) {
	for (int i = 0; i < n; i++) {
		memcpy (out_relation + i * num_tuples, in_relation, num_tuples * sizeof(int));
	}

	/*unsigned short state[3] = {0, 0, 0};
    unsigned int seed       = time(NULL);
    memcpy(state, &seed, sizeof(seed));

	knuth_shuffle48(out_relation, num_tuples * n, state);*/

	return 0;
}

/**
 * Generate tuple IDs -> random distribution
 * relation must have been allocated
 */
void random_gen(int *rel, uint64_t elsNum, const int64_t maxid) {
	uint64_t i;

	for (i = 0; i < elsNum; i++) {
		rel[i] = RAND_RANGE(maxid);
//		printf("%d: rel[%d] = %d\n", maxid, i, rel[i]);
	}
}

/**
 * Create random unique keys starting from firstkey
 */
void random_unique_gen(int *rel, uint64_t elsNum, const int64_t maxid) {
    uint64_t i;

    uint64_t firstkey = 0;

    /* for randomly seeding nrand48() */
    unsigned short state[3] = {0, 0, 0};
    unsigned int seed       = time(NULL);
    memcpy(state, &seed, sizeof(seed));

    for (i = 0; i < elsNum; i++) {
        rel[i] = firstkey;

        if(firstkey == maxid)
            firstkey = 0;

        firstkey ++;
    }

    /* randomly shuffle elements */
    knuth_shuffle48(rel, elsNum, state);

   }

/**
 * Create a foreign-key relation using the given primary-key relation and
 * foreign-key relation size. Keys in pkrel is randomly distributed in the full
 * integer range.
 *
 * @param fkrel [output] foreign-key relation
 * @param pkrel [input] primary-key relation
 * @param num_tuples
 *
 * @return
 */
int create_relation_fk_from_pk(const char *fkrelFilename, int *fkrel, uint64_t fkrelElsNum, int *pkrel,
		uint64_t pkrelElsNum) {
	/*first try to read from a file*/
	if (readFromFile(fkrelFilename, fkrel, fkrelElsNum)) {
		int i, iters;
		int64_t remainder;

		/* alternative generation method */
		iters = fkrelElsNum / pkrelElsNum;
		for (i = 0; i < iters; i++) {
			memcpy(fkrel + i * pkrelElsNum, pkrel, pkrelElsNum * sizeof(int));
		}

		/* if num_tuples is not an exact multiple of pkrel->num_tuples */
		remainder = fkrelElsNum % pkrelElsNum;
		if (remainder > 0) {
			memcpy(fkrel + i * pkrelElsNum, pkrel, remainder * sizeof(int));
		}

		knuth_shuffle(fkrel, fkrelElsNum);

		return writeToFile(fkrelFilename, fkrel, fkrelElsNum);
	}

	return 0;
}

/**
 * Shuffle tuples of the relation using Knuth shuffle.
 *
 * @param relation
 */
void knuth_shuffle(int *relation, uint64_t elsNum) {
	int64_t i;
	for (i = elsNum - 1; i > 0; i--) {
		int64_t j = RAND_RANGE(i);
		int tmp = relation[i];
		relation[i] = relation[j];
		relation[j] = tmp;
	}
}

void knuth_shuffle48(int *relation, uint64_t elsNum, unsigned short * state) {
	int64_t i;
	for (i = elsNum - 1; i > 0; i--) {
		int64_t j = RAND_RANGE48(i, state);
		int tmp = relation[i];
		relation[i] = relation[j];
		relation[j] = tmp;
	}
}

int create_relation_zipf(const char *filename, int *relation, uint64_t elsNum, const int64_t maxid,
		const double zipf_param) {
	/*first try to read from a file*/
	if (readFromFile(filename, relation, elsNum)) {
		check_seed();

		gen_zipf(elsNum, maxid, zipf_param, relation);

		return writeToFile(filename, relation, elsNum);
	}
	return 0;
}

/**
 * Create an alphabet, an array of size @a size with randomly
 * permuted values 0..size-1.
 *
 * @param size alphabet size
 * @return an <code>item_t</code> array with @a size elements;
 *         contains values 0..size-1 in a random permutation; the
 *         return value is malloc'ed, don't forget to free it afterward.
 */
static uint32_t *gen_alphabet(unsigned int size) {
	uint32_t *alphabet;

	/* allocate */
	alphabet = (uint32_t *) malloc(size * sizeof(*alphabet));
	assert(alphabet);

	/* populate */
	for (unsigned int i = 0; i < size; i++)
		alphabet[i] = i + 1; /* don't let 0 be in the alphabet */

	/* permute */
	for (unsigned int i = size - 1; i > 0; i--) {
		unsigned int k = (unsigned long) i * rand() / RAND_MAX;
		unsigned int tmp;

		tmp = alphabet[i];
		alphabet[i] = alphabet[k];
		alphabet[k] = tmp;
	}

	return alphabet;
}

/**
 * Generate a lookup table with the cumulated density function
 *
 * (This is derived from code originally written by Rene Mueller.)
 */
static double *gen_zipf_lut(double zipf_factor, unsigned int alphabet_size) {
	double scaling_factor;
	double sum;

	double *lut; /**< return value */

	lut = (double *) malloc(alphabet_size * sizeof(*lut));
	assert(lut);

	/*
	 * Compute scaling factor such that
	 *
	 *   sum (lut[i], i=1..alphabet_size) = 1.0
	 *
	 */
	scaling_factor = 0.0;
	for (unsigned int i = 1; i <= alphabet_size; i++)
		scaling_factor += 1.0 / pow(i, zipf_factor);

	/*
	 * Generate the lookup table
	 */
	sum = 0.0;
	for (unsigned int i = 1; i <= alphabet_size; i++) {
		sum += 1.0 / pow(i, zipf_factor);
		lut[i - 1] = sum / scaling_factor;
	}

	return lut;
}

/**
 * Generate a stream with Zipf-distributed content.
 */
void gen_zipf(uint64_t stream_size, unsigned int alphabet_size, double zipf_factor, int *ret) {
	//uint64_t i;
	/* prepare stuff for Zipf generation */
	uint32_t *alphabet = gen_alphabet(alphabet_size);
	assert(alphabet);

	double *lut = gen_zipf_lut(zipf_factor, alphabet_size);
	assert(lut);

	uint32_t seeds[64];

	for (int i = 0; i < 64; i++)
		seeds[i] = rand();

	for (uint64_t i = 0; i < stream_size; i++) {
		if (i % 1000000 == 0)
			printf("live %d\n", i / 1000000);

		/* take random number */
		double r;

		r = ((double) (rand())) / RAND_MAX;

		/* binary search in lookup table to determine item */
		unsigned int left = 0;
		unsigned int right = alphabet_size - 1;
		unsigned int m; /* middle between left and right */
		unsigned int pos; /* position to take */

		if (lut[0] >= r)
			pos = 0;
		else {
			while (right - left > 1) {
				m = (left + right) / 2;

				if (lut[m] < r)
					left = m;
				else
					right = m;
			}

			pos = right;
		}

		ret[i] = alphabet[pos];
	}

	free(lut);
	free(alphabet);
}

