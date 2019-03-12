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

#include "common-host.h"

#include <sys/time.h>

double cpuSeconds() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void initializeSeq(int *in, size_t size)	{
	for(int i = 0 ; i < size; i++)	{
		in[i] = i;
	}
}

void initializeUniform(int *in, size_t size)	{
	//srand (time(NULL));
	//We want the input to be the same for every test
	//BUT: If seed is set to 1, the generator is reinitialized to
	//its initial value and produces the same values
	//as before any call to rand or srand.
//	srand (1);
//	for(int i = 0 ; i < size; i++)	{
//		in[i] = rand() % size;
//	}

	for (int i = 0; i < size; i++) {
		in[i] = rand() % size;
	}
}

void initializeUniform(int *in, size_t size, int seed) {
//	//srand (time(NULL));
//	//We want the input to be the same for every test
//	//BUT: If seed is set to 1, the generator is reinitialized to
//	//its initial value and produces the same values
//	//as before any call to rand or srand.
//	srand(seed + 10);
//	for(int i = 0 ; i < size; i++)	{
//		in[i] = rand() % size;
//	}

	struct random_data* rand_states;
	char* rand_statebufs;
	int nthreads = 1;
	int bufferSize = 32;
	rand_states = (struct random_data*) calloc(nthreads,
			sizeof(struct random_data));
	rand_statebufs = (char*) calloc(nthreads, bufferSize);

	/* for each 'thread', initialize a PRNG (the seed is the first argument) */
	//initstate_r(random(), &rand_statebufs[t], PRNG_BUFSZ, &rand_states[t]);
	initstate_r(seed + 10, &rand_statebufs[0], bufferSize, &rand_states[0]);
	int state1;

	for (int i = 0; i < size; i++) {
		random_r(&rand_states[0], &state1);
		in[i] = state1 % size;
	}

	free(rand_states);
	free(rand_statebufs);
}

void initializeUniform(int *in, size_t size, int maxNo, int seed) {
	//srand (time(NULL));
	//We want the input to be the same for every test
	//BUT: If seed is set to 1, the generator is reinitialized to
	//its initial value and produces the same values
	//as before any call to rand or srand.
//	srand(seed + 10);
//	for(int i = 0 ; i < size; i++)	{
//		in[i] = rand() % maxNo;
//	}

	struct random_data* rand_states;
	char* rand_statebufs;
	int nthreads = 1;
	int bufferSize = 32;
	rand_states = (struct random_data*) calloc(nthreads,
			sizeof(struct random_data));
	rand_statebufs = (char*) calloc(nthreads, bufferSize);

	/* for each 'thread', initialize a PRNG (the seed is the first argument) */
	//initstate_r(random(), &rand_statebufs[t], PRNG_BUFSZ, &rand_states[t]);
	initstate_r(seed + 10, &rand_statebufs[0], bufferSize, &rand_states[0]);
	int state1;
	for (int i = 0; i < size; i++) {
		random_r(&rand_states[0], &state1);
		in[i] = state1 % maxNo;
	}

	free(rand_states);
	free(rand_statebufs);
}

void initializeZero(int *in, size_t size)	{
	for(int i = 0 ; i < size; i++)	{
		in[i] = 0;
	}
}

int NumberOfSetBits(int i) //uint32_t
{
     // Java: use >>> instead of >>
     // C or C++: use uint32_t
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}
