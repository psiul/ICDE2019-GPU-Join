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

#include "common.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>           /* gettimeofday */
#include <math.h>

__constant__ unsigned int valuesToProcess;
__device__ maxSize_type sum_dev;


void recordTime(time_st *t) {
	gettimeofday(t, NULL);
}

unsigned int smallestGreaterPowerOf2(const unsigned int num) {
	unsigned int x = (UINT_MAX >> 1) + 1; //the greatest possible power of 2
	while (!(x & num))
		x >>= 1;
	if (x ^ num) return x << 1;
	return x; /*size is already a power of 2*/

}

unsigned int greatestLowerPowerOf2(const unsigned int num) {
	unsigned int x = (UINT_MAX >> 1) + 1; //the greatest possible power of 2
	while (!(x & num))
		x >>= 1;
	if (x ^ num) return x;
	return x >> 1; /*size is already a power of 2*/
}

void initialise_float(float *A, int N) {
	int i;
	for (i = 0; i < N; i++)
		A[i] = (float) (rand() & 0xff) / 10.0f;
}

void initialise_int(int *A, const int N) {
	int i;
	for (i = 0; i < N; i++)
		A[i] = rand() % 100 + 1;
}

void printArray_int(int *A, const size_t N) {
	if (N > (1 << 10)) return;
	int i;
	for (i = 0; i < N; i++) {
		printf("%5d", A[i]);
		if ((i + 1) % 35 == 0) printf("\n");
	}
	printf("\n");
}

void printArray_uint(unsigned int *A, const size_t N) {
	if (N > (1 << 22)) return;
	int i;
	for (i = 0; i < N; i++) {
		printf("%6u", A[i]);
		if ((i + 1) % 35 == 0) printf("\n");
	}
	printf("\n");
}

void printArray_maxSize_type(maxSize_type *A, const maxSize_type N) {
	if (N > (1 << 10)) return;
	int i;
	for (i = 0; i < N; i++) {
		printf("%5lu", A[i]);
		if ((i + 1) % 35 == 0) printf("\n");
	}
	printf("\n");
}

void printArray_char(char *A, const maxSize_type N) {
	if (N > (1 << 22)) return;
	maxSize_type i;
	for (i = 0; i < N; i++) {
		printf("%6d", A[i]);
		if ((i + 1) % 30 == 0) printf("\n");
	}
	printf("\n");
}

__global__ void copy(data_type *dataTO, data_type *dataFROM, const maxSize_type size) {
	maxSize_type gidx = blockIdx.x*blockDim.x + threadIdx.x;
	if(gidx >= size) return;
	dataTO[gidx] = dataFROM[gidx];
}

__global__ void copy(maxSize_type *dataTO, maxSize_type *dataFROM, const maxSize_type size) {
	maxSize_type gidx = blockIdx.x*blockDim.x + threadIdx.x;
	if(gidx >= size) return;
	dataTO[gidx] = dataFROM[gidx];
}

__global__ void scatter(data_type *dataIN, data_type *dataOUT, const maxSize_type size, maxSize_type *pos) {
	uint64_t gidx = blockIdx.x*blockDim.x + threadIdx.x;
	if(gidx >= size) return;
//	if(pos[gidx] > 1010) {
//		printf("(%d,%d) : data[%lu] <- %d\n", blockIdx.x, threadIdx.x, pos[gidx], dataIN[gidx]);
//	}
	dataOUT[pos[gidx]] = dataIN[gidx];
}

/*Processes at most threadsNum elements of type uint64_t*/
__device__ void prefixSum_after(maxSize_type *data, const unsigned int size, maxSize_type *total) {
	unsigned int idx = threadIdx.x, idx_f, idx_s;

	/*iterate until the final result is computed */
	unsigned int stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) data[idx_f] += data[idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	*total = data[0]; //all threads get the result;
	__syncthreads();

	/*store the final results*/
	if (threadIdx.x == 0) {
		data[0] = 0;
	}
	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) {
			int tmp = data[idx_s];
			data[idx_s] = data[idx_f];
			data[idx_f] = tmp + data[idx_s];
		}
		__syncthreads();
	}
}

/*Processes at most threadsNum elements of type maxSize_type*/
__device__ void prefixSum_before(maxSize_type *data, const unsigned int size, maxSize_type *total) {
	unsigned int idx = threadIdx.x;
	unsigned int idx_f, idx_s;


	unsigned int stride;

	/*iterate until the final result is computed */
	for (stride = 1; stride < size; stride <<= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) data[size - 1 - idx_f] += data[size - 1 - idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	*total = data[size-1];

	__syncthreads();

	/*store the final results*/
	if (threadIdx.x == 0) {
		data[size - 1] = 0;
//		printf("*total = %lu\n", *total);
	}

	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) {
			maxSize_type tmp = data[size - 1 - idx_s];
			data[size - 1 - idx_s] = data[size - 1 - idx_f];
			data[size - 1 - idx_f] = tmp + data[size - 1 - idx_s];
		}
		__syncthreads();
	}
}

/*Processes at most threadsNum elements of type data_type*/
__device__ void prefixSum_before(data_type *const data, const unsigned int size, data_type *total) {
	unsigned int idx = threadIdx.x;
	unsigned int idx_f, idx_s;


	unsigned int stride;

	/*iterate until the final result is computed */
	for (stride = 1; stride < size; stride <<= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) data[size - 1 - idx_f] += data[size - 1 - idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	*total = data[size-1];

	__syncthreads();

	/*store the final results*/
	if (threadIdx.x == 0) {
		data[size - 1] = 0;
//		printf("*total = %lu\n", *total);
	}

	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) {
			maxSize_type tmp = data[size - 1 - idx_s];
			data[size - 1 - idx_s] = data[size - 1 - idx_f];
			data[size - 1 - idx_f] = tmp + data[size - 1 - idx_s];
		}
		__syncthreads();
	}
}

__device__ void prefixSum_before_multiple(data_type *const data, const unsigned int size, data_type *total, unsigned int num) {
	unsigned int idx = threadIdx.x;
	unsigned int idx_f, idx_s;
	unsigned int i;

	unsigned int stride;

	for (i = 0; i < num; i++) {
		data_type *data_local = data + i * size;
//		printf("%d/%d (%d,%d) : data_local [%p, %p]\n", i, num, blockIdx.x, threadIdx.x, data_local, data_local+size);

		/*iterate until the final result is computed */
		for (stride = 1; stride < size; stride <<= 1) {
			idx_f = stride * (2 * idx);
			idx_s = stride * (2 * idx + 1);

			if (idx_s < size) {
//				if(data_local + size - 1 - idx_f == addr)
//				printf("%d/%d U (%d,%d) -> data_local[%d] (%p) = %d, data_local[%d] (%p) = %d\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, data_local + size - 1 - idx_f,
//						data_local[size - 1 - idx_f], size - 1 - idx_s, data_local + size - 1 - idx_s, data_local[size - 1 - idx_s]);
				data_local[size - 1 - idx_f] += data_local[size - 1 - idx_s];
//				if(data_local + size - 1 - idx_f == addr) printf("%d/%d U (%d,%d) -> data_local[%d] = %d\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, data_local[size - 1 - idx_f]);
			}

			/*wait for all the threads in the block to finish before going to the next iteration*/
			__syncthreads();
		}

		total[i] = data_local[size - 1];
//		printf("%u : (%d,%d) -> total[%d] = %d\n", size, blockIdx.x, threadIdx.x, i, total[i]);

		__syncthreads();
//	}

//	/*store the final results (assumes more threads than num)*/
//	if (threadIdx.x < num) {
//		(data+threadIdx.x*size)[size - 1] = 0;
////		printf("(%d,%d) : total[%d] = %d\n", blockIdx.x, threadIdx.x, threadIdx.x, total[threadIdx.x]);
//	}
//
//	__syncthreads();

//	for (i = 0; i < num; i++) {
//		data_type *data_local = data + i * size;
//		if(blockIdx.x == 0) printf("%d/%d\n", i,num);

		if (threadIdx.x == 0) data_local[size - 1] = 0;
		__syncthreads();

//		printf("%p\n", addr);
//		if(data_local + threadIdx.x == addr)
//			printf("%d/%d D0 (%d,%d) -> (%p) data_local[%d] = %d\n", i, num, blockIdx.x, threadIdx.x, data_local + threadIdx.x, threadIdx.x, data_local[threadIdx.x]);

		/*now go the other direction*/
		for (stride >>= 1; stride > 0; stride >>= 1) {
			idx_f = stride * (2 * idx);
			idx_s = stride * (2 * idx + 1);

//			if(blockIdx.x == 0)	printf("%d/%d D0 (%d,%d) -> data_local[%d], data_local[%d]\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, size - 1 - idx_s);

			if (idx_s < size) {
//				if(data_local + size - 1 - idx_f == addr)
//				printf("%d/%d D0 (%d,%d) -> (%p) data_local[%d] = %d, (%p) data_local[%d] = %d\n", i, num, blockIdx.x, threadIdx.x, data_local + size - 1 - idx_f, size - 1 - idx_f,
//						data_local[size - 1 - idx_f], data_local + size - 1 - idx_s, size - 1 - idx_s, data_local[size - 1 - idx_s]);
//				data_type tmps = data_local[size - 1 - idx_s];
//				data_type tmpf = data_local[size - 1 - idx_f];
				data_type tmp = data_local[size - 1 - idx_s];
//				if (data_local + size - 1 - idx_f == addr)
//				printf("%d/%d D1 (%d,%d) -> data_local[%d] = %d, data_local[%d] = %d, %d\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, data_local[size - 1 - idx_f], size - 1 - idx_s,
//						data_local[size - 1 - idx_s], tmp);
				data_local[size - 1 - idx_s] = data_local[size - 1 - idx_f];
//				data_local[size - 1 - idx_f] = tmpf + tmps;
//				if (data_local + size - 1 - idx_f == addr)
//				printf("%d/%d D2 (%d,%d) -> data_local[%d] = %d, data_local[%d] = %d, %d\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, data_local[size - 1 - idx_f], size - 1 - idx_s,
//						data_local[size - 1 - idx_s], tmp);
				data_local[size - 1 - idx_f] = tmp + data_local[size - 1 - idx_s];
//				data_local[size - 1 - idx_s] = tmpf;
//				if (data_local + size - 1 - idx_f == addr)
//				printf("%d/%d D (%d,%d) -> data_local[%d] = %d, data_local[%d] = %d, %d\n", i, num, blockIdx.x, threadIdx.x, size - 1 - idx_f, data_local[size - 1 - idx_f], size - 1 - idx_s,
//						data_local[size - 1 - idx_s], tmp);
			}
			__syncthreads();
		}

		__syncthreads();
	}
}

/*Processes at most threadsNum elements of type uint64_t. Write the total to the sum_dev*/
__device__ void prefixSum_before_device(maxSize_type *data, const unsigned int size) {
	unsigned int idx = threadIdx.x;
	unsigned int idx_f, idx_s;
	unsigned int stride;

//	if(threadIdx.x == 0) printf("(%d,%d) : %u\n", blockIdx.x, threadIdx.x, size);

//	printf("B: %u data[%u] = %lu (%u,%u)\n", blockIdx.x, idx, data[idx], stride, size);

	/*iterate until the final result is computed */
	for (stride = 1; stride < size; stride <<= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

//		if (idx_s < size) printf("%u (%d,%d) : %u, %u\n", stride, blockIdx.x, threadIdx.x, size - 1 - idx_f, size - 1 - idx_s);

		if (idx_s < size) data[size - 1 - idx_f] += data[size - 1 - idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

//	printf("A: %u data[%u] = %lu\n", blockIdx.x, idx, data[idx]);

	/*store the final results*/
	if (threadIdx.x == 0) {
		atomicAdd(&sum_dev, data[size-1]);
		data[size - 1] = 0;
	}

	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) {
			maxSize_type tmp = data[size - 1 - idx_s];
			data[size - 1 - idx_s] = data[size - 1 - idx_f];
			data[size - 1 - idx_f] = tmp + data[size - 1 - idx_s];
		}
		__syncthreads();
	}
}

__device__ void prefixSum_before_multipleSeq(data_type *const data, const unsigned int size, data_type *const borders, const unsigned int bordersNum, data_type *totalPerSeq) {
	/*set the ids and the data according to the ranges*/
	unsigned int lidx = threadIdx.x;
	data_type range = data[lidx];
	data_type border = borders[range];

	lidx -= border;

	data_type *data_local = data+border;
	data_type *totalPerSeq_local = totalPerSeq + range;

	unsigned int size_local = 0;
	if(range == 0) size_local = borders[range+1];
	else if(range == bordersNum-1) size_local = size-border;
	else size_local = borders[range+1]-border;

	if(threadIdx.x < bordersNum)
		totalPerSeq[threadIdx.x] = 0; //initialise

	__syncthreads();

//	printf("(%d,%d) : range = %d, border = %d -> idx = %d, size = %d, data = %d\n", blockIdx.x, threadIdx.x, range, border, lidx, size_local, data_local[lidx]);
	/*continue with prefixSum as usual*/
	unsigned int idx_f, idx_s;
	unsigned int stride;

	/*iterate until the final result is computed */
	for (stride = 1; stride < size_local; stride <<= 1) {
		idx_f = stride * (2 * lidx);
		idx_s = stride * (2 * lidx + 1);

		if (idx_s < size_local) data_local[size_local - 1 - idx_f] += data_local[size_local - 1 - idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	*totalPerSeq_local = data_local[size_local-1];
//	printf("(%d,%d) : total = %d (%d)\n", blockIdx.x, threadIdx.x, *totalPerSeq_local, range);

	__syncthreads();

	/*store the final results*/
	if (lidx == 0) {
		data_local[size_local - 1] = 0;

	}

	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * lidx);
		idx_s = stride * (2 * lidx + 1);

		if (idx_s < size_local) {
			maxSize_type tmp = data_local[size_local - 1 - idx_s];
			data_local[size_local - 1 - idx_s] = data_local[size_local - 1 - idx_f];
			data_local[size_local - 1 - idx_f] = tmp + data_local[size_local - 1 - idx_s];
		}
		__syncthreads();
	}

//	printf("(%d,%d) : data = %d\n", blockIdx.x, threadIdx.x, data_local[lidx]);
}


__global__ void prefixSum_before(maxSize_type *data, const maxSize_type size, maxSize_type *total) {
	if(size <= blockDim.x*blockIdx.x) return; /*nothing for this block*/

	maxSize_type gidx = blockIdx.x*blockDim.x+threadIdx.x;
	maxSize_type blockTotal;

	/*write data to shared memory*/
	if(gidx < size) uint64_shared[threadIdx.x] = data[gidx];

	__syncthreads(); /*wait for all threads to finish writing in shared memory*/

	unsigned int blockSize = blockDim.x;
	if(blockSize > size - blockDim.x*blockIdx.x) blockSize = size - blockDim.x*blockIdx.x;

	prefixSum_before(uint64_shared, blockSize, &blockTotal);

	/*write the results back from shared memory*/
	if(gidx < size) data[gidx] = uint64_shared[threadIdx.x];

	/*all threads in the block get the blockTotal but only one writes it in the output*/
	if(threadIdx.x == 0) *(total+blockIdx.x) = blockTotal;
}

/*there is no total per block, there is only the sum of all totals of all blocks
 * If there is only one block then the sum of all totals is the total sum of the block.
 * The kernel does not return the total. it just computes it and stores it in case another
 * kernel wants to use it later*/
__global__ void prefixSum_before_device(maxSize_type *data, const maxSize_type size) {

	if(size <= blockDim.x*blockIdx.x) return; /*nothing for this block*/

	maxSize_type gidx = blockIdx.x*blockDim.x+threadIdx.x;

	if(threadIdx.x == 0) sum_dev = 0; //initialise the sum to clean from any old result

	/*write data to shared memory*/
	if(gidx < size) uint64_shared[threadIdx.x] = data[gidx];

	__syncthreads(); /*wait for all threads to finish writing in shared memory*/

	unsigned int blockSize = blockDim.x;
	if(blockSize > size - blockDim.x*blockIdx.x) blockSize = size - blockDim.x*blockIdx.x;

	prefixSum_before_device(uint64_shared, blockSize);

	/*write the results back from shared memory*/
	if(gidx < size) data[gidx] = uint64_shared[threadIdx.x];

//	if(threadIdx.x == 0) printf("%d : blockSum = %lu\n", blockIdx.x, sum_dev);
}

__device__ void sum(maxSize_type *data, const unsigned int size, maxSize_type *res) {
	/*iterate until the final result is computed */
	maxSize_type stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		maxSize_type idx_f = stride * (2 * threadIdx.x);
		maxSize_type idx_s = stride * (2 * threadIdx.x + 1);

//		if(idx_f < size)
//			printf("(%d, %d) : data[%lu] = %lu\n", blockIdx.x, threadIdx.x, idx_f, data[idx_f]);
//		if(idx_s < size)
//			printf("(%d, %d) : data[%lu] = %lu\n", blockIdx.x, threadIdx.x, idx_s, data[idx_s]);

		if (idx_s < size) data[idx_f] += data[idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	/*store the final results*/
	*res = data[0];
//	if (threadIdx.x == 0)	printf("(%d,%d): %u %lu - %lu\n", blockIdx.x, threadIdx.x, size, *res, data[0]);
}

__device__ void sum_device(maxSize_type *data, const unsigned int size) {
	/*iterate until the final result is computed */
	maxSize_type stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		maxSize_type idx_f = stride * (2 * threadIdx.x);
		maxSize_type idx_s = stride * (2 * threadIdx.x + 1);

//		if(idx_f < size)
//			printf("(%d, %d) : data[%lu] = %lu\n", blockIdx.x, threadIdx.x, idx_f, data[idx_f]);
//		if(idx_s < size)
//			printf("(%d, %d) : data[%lu] = %lu\n", blockIdx.x, threadIdx.x, idx_s, data[idx_s]);

		if (idx_s < size) data[idx_f] += data[idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	/*store the final results*/
	if(threadIdx.x == 0) sum_dev = data[0];
//	if (threadIdx.x == 0)	printf("(%d,%d): %u %lu - %lu\n", blockIdx.x, threadIdx.x, size, *res, data[0]);
}

__global__ void sum(maxSize_type *data, const maxSize_type size, maxSize_type *res) {
	if(size <= blockDim.x*blockIdx.x) return; /*nothing for this block*/

	maxSize_type gidx = blockIdx.x*blockDim.x + threadIdx.x;

	maxSize_type total;

	/*write data to shared memory*/
	if(gidx < size) uint64_shared[threadIdx.x] = data[gidx];

	__syncthreads();

	unsigned int blockSize = blockDim.x;
	if (blockSize > size - blockDim.x * blockIdx.x) blockSize = size - blockDim.x * blockIdx.x;
	sum(uint64_shared, blockSize, &total);

	/*I want to write to output only once. For that I use a
	 * local variable for the local and then have only thread 0
	 * write to output
	 */
	if (threadIdx.x == 0)
		*res = total;
}

__global__ void sum_device(maxSize_type *data, const maxSize_type size) {
	if(size <= blockDim.x*blockIdx.x) return; /*nothing for this block*/

	maxSize_type gidx = blockIdx.x*blockDim.x + threadIdx.x;

	/*write data to shared memory*/
	if(gidx < size) uint64_shared[threadIdx.x] = data[gidx];
//	printf("(%d,%d) : data[%lu] = %lu\n", blockIdx.x, threadIdx.x, gidx, data[gidx]);

	__syncthreads();

	unsigned int blockSize = blockDim.x;
	if (blockSize > size - blockDim.x * blockIdx.x) blockSize = size - blockDim.x * blockIdx.x;
	sum_device(uint64_shared, blockSize);
}


/*I assume threads equal to the total number of elements per range. Data are already at shared memory*/
__device__ void prefixSum_sharedMem_before_multipleRanges(maxSize_type *data, maxSize_type data_els, maxSize_type *total1,
		maxSize_type *total2, maxSize_type *total3, unsigned int partitionsNum) {
	maxSize_type idx_f, idx_s;
	maxSize_type gidx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int lidx = threadIdx.x;
	unsigned int size = blockDim.x;
	maxSize_type stride;
	unsigned int i;

//	if(gidx >= data_els) return; /* I want all  threads to terminate so that all of them can have the total value*/

	/*find exactly how many elements this block has in shared memory*/
	if (size > data_els - blockIdx.x * blockDim.x) size = data_els - blockIdx.x * blockDim.x;

//	printf("\n(%d,%d) : shared[%u] = data[%lu] = %lu", blockIdx.x, lidx, lidx, gidx, uint64_shared[lidx]);

	maxSize_type *data_ptr;

	/*iterate until the final result is computed */
	for (stride = 1; stride < size; stride <<= 1) {
		idx_f = stride * (2 * lidx);
		idx_s = stride * (2 * lidx + 1);

		data_ptr = data;
		for (i = 0; i < partitionsNum; i++) {

//		printf("\n%lu (%d,%d) : %lu, %lu -> %lu, %lu", stride, blockIdx.x, lidx, idx_f, idx_s, size - 1 - idx_f,size - 1 - idx_s);

			if (idx_s < size) {
//			uint64_t tmp = uint64_shared[size - 1 - idx_f];
				data_ptr[size - 1 - idx_f] += data_ptr[size - 1 - idx_s];
//			printf("\nUP %lu (%d,%d) : shared[%u] = %lu+%lu=%lu", stride, blockIdx.x, lidx, size - 1 - idx_f, tmp, uint64_shared[size - 1 - idx_s], uint64_shared[size - 1 - idx_f]);
			}
			data_ptr += blockDim.x;
		}

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	/*store the final results*/
//	if (threadIdx.x == 0) {
	*total1 = data[size - 1];
	*total2 = data[blockDim.x + size - 1];
	*total3 = data[2 * blockDim.x + size - 1];

	__syncthreads();
//printf("\n-> (%d,%d) : %d", blockIdx.x, threadIdx.x, *total);
	if (threadIdx.x == 0) {
		for (i = 0; i < partitionsNum; i++)
			data[i * blockDim.x + size - 1] = 0;
	}

	__syncthreads();

	/*now go the other direction*/
	for (stride >>= 1; stride > 0; stride >>= 1) {
		idx_f = stride * (2 * lidx);
		idx_s = stride * (2 * lidx + 1);

		data_ptr = data;
		for (i = 0; i < partitionsNum; i++) {

			if (idx_s < size) {
				maxSize_type tmp = data_ptr[size - 1 - idx_s];
				data_ptr[size - 1 - idx_s] = data_ptr[size - 1 - idx_f];
				data_ptr[size - 1 - idx_f] = tmp + data_ptr[size - 1 - idx_s];
//			printf("\nDOWN %lu (%d,%d) : shared[%u] = %lu -  shared[%u] = %lu", stride, blockIdx.x, lidx, size - 1 - idx_f, uint64_shared[size - 1 - idx_f], size - 1 - idx_s, uint64_shared[size - 1 - idx_s]);
			}
			data_ptr += blockDim.x;
		}
		__syncthreads();
	}
}

__device__ void sum(int *data, unsigned int size, int *res) {
	unsigned int idx, idx_f, idx_s;

	/*iterate until the final result is computed */
	int stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		idx = threadIdx.x;

		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		if (idx_s < size) data[idx_f] += data[idx_s];

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

//	/*store the final results*/
//	if (threadIdx.x == 0) {
	*res = data[0];
//	}
}



__device__ void max(int *data, unsigned int size, int *res) {
	unsigned int idx, idx_f, idx_s;

	/*iterate until the final result is computed */
	int stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		idx = threadIdx.x;

		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		while (idx_s < size) {
//			int tmp = data[idx_f];
			if (data[idx_f] < data[idx_s]) data[idx_f] = data[idx_s];
//			printf("UP %d (%d,%d): %d [%d]=[%d]+[%d]=%d+%d=%d\n", size, blockIdx.x, threadIdx.x, stride, idx_f, idx_f, idx_s, tmp, data[idx_s], data[idx_f]);

			idx += blockDim.x;

			idx_f = stride * (2 * idx);
			idx_s = stride * (2 * idx + 1);
		}

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	/*store the final results*/
	if (threadIdx.x == 0) {
		*res = data[0];
//	printf("%d: Max = %d\n", blockIdx.x, *res);
	}
}

__device__ void min(int *data, unsigned int size, int *res) {
	unsigned int idx, idx_f, idx_s;

	/*iterate until the final result is computed */
	int stride = 1;
	for (stride = 1; stride < size; stride <<= 1) {
		idx = threadIdx.x;

		idx_f = stride * (2 * idx);
		idx_s = stride * (2 * idx + 1);

		while (idx_s < size) {
//			int tmp = data[idx_f];
			if (data[idx_f] > data[idx_s]) data[idx_f] = data[idx_s];
//			printf("UP %d (%d,%d): %d [%d]=[%d]+[%d]=%d+%d=%d\n", size, blockIdx.x, threadIdx.x, stride, idx_f, idx_f, idx_s, tmp, data[idx_s], data[idx_f]);

			idx += blockDim.x;

			idx_f = stride * (2 * idx);
			idx_s = stride * (2 * idx + 1);
		}

		/*wait for all the threads in the block to finish before going to the next iteration*/
		__syncthreads();
	}

	/*store the final results*/
	if (threadIdx.x == 0) {
		*res = data[0];
//	printf("%d: Min = %d\n", blockIdx.x, *res);
	}
}

__global__ void aggregate(int *data, unsigned int size, int *res, int funcId) {
	/*all processing should be done in the same block*/
	if (blockIdx.x > 0) return;

	unsigned int idx;
	for (idx = threadIdx.x; idx < size; idx += blockDim.x)
		int_shared[idx] = data[idx];

	__syncthreads();

	if (funcId == 1)
		min(int_shared, size, res);
	else if (funcId == 2)
		max(int_shared, size, res);
	else if (funcId == 3) sum(int_shared, size, res);
}

void* test(size_t sz) {
	//CUDA UVA

	void* mem;
	cudaHostAlloc((void **) &mem, sz, cudaHostAllocMapped);

	if (mem)
		return mem;
	else
		return NULL;
}

#define PREFIX_SUM2
#define SUM

static __global__ void addWithStepANDsum_device(maxSize_type *data, maxSize_type size, maxSize_type *total, maxSize_type total_els, maxSize_type group) {
	maxSize_type gidx = blockDim.x * blockIdx.x + threadIdx.x;

	if (blockIdx.x <= group) return;

//	if(threadIdx.x == 0) printf("(%d,%d) : blockSum = %lu\n", blockIdx.x, threadIdx.x, sum_dev);

	unsigned int blockIdx_normalised = blockIdx.x - group;
#if defined(SUM) || defined(PREFIX_SUM)
	if (blockIdx_normalised >= total_els && gidx < size) {
		/*use the pre-computed sum*/
//		if(gidx == 20) printf("(%d,%d) : blockSum = %lu\n", blockIdx.x, threadIdx.x, sum_dev);
		data[gidx] += sum_dev;
	} else {
#endif
		/*use the pre-computed prefixSum*/
//		if(gidx == 20) printf("(%d,%d) : blockSum = %lu\n", blockIdx.x, threadIdx.x, total[blockIdx_normalised]);
#if defined(PREFIX_SUM)
		if(gidx < size) data[gidx] += total[blockIdx_normalised];
#else
		if (threadIdx.x < blockIdx_normalised && threadIdx.x < total_els)
			uint64_shared[threadIdx.x] = total[threadIdx.x];

//		if(threadIdx.x == 0) printf("(%d,%d) : shared memory\n", blockIdx.x, threadIdx.x);

		__syncthreads();

		maxSize_type sumOfAll;
		unsigned int elsNum = blockDim.x;
		if (blockIdx.x - group < blockDim.x) elsNum = blockIdx.x - group;

		sum(uint64_shared, elsNum, &sumOfAll);

//		if(gidx == 4)
//			printf("(%d,%d) : %lu elsNum = %lu, sum = %lu\n", blockIdx.x, threadIdx.x, gidx, elsNum, sumOfAll);

		if(gidx < size) data[gidx] += sumOfAll;
#endif
#if defined(SUM) || defined(PREFIX_SUM)
	}
#endif
}

void totalPrefixSum(maxSize_type *data, maxSize_type size, maxSize_type *total, maxSize_type *sumOfAll, uint threadsNum, unsigned int iterNum) {
	unsigned int blocksNum = (size + threadsNum - 1) / threadsNum;
	dim3 block(threadsNum);
	dim3 grid(blocksNum);
	unsigned int sharedMemSize = threadsNum * sizeof(maxSize_type); /*one positions for each thread*/

	unsigned int i = 0;
	maxSize_type s = data[size - 1];

//printf("%u --> s: %u d: %u - %u t: %u - %u\n", iterNum, sumOfAll, data, data+size, total, total+block.x);
	/*compute first prefix sum*/
		prefixSum_before<<<grid,block, sharedMemSize, 0>>>(data, size, total);
	
	
	if (blocksNum > threadsNum) {
		for (i = 0; i < blocksNum - threadsNum; i += threadsNum) {
//			printf("Adding %lu (%d, %d)\n", i, blocksNum, threadsNum);
			/*compute the sum of as many elements as the number of threads*/
#if defined(PREFIX_SUM)
			prefixSum_before_device<<<grid, block, sharedMemSize,0>>>(total+i, (maxSize_type)threadsNum);
#elif defined(SUM)
			sum_device<<<grid, block, sharedMemSize,0>>>(total+i, (maxSize_type)threadsNum);
#endif
//			CHK_ERROR(cudaDeviceSynchronize());
			addWithStepANDsum_device<<<grid,block,sharedMemSize,0>>>(data, size, total+i, threadsNum, i);
			CHK_ERROR(cudaDeviceSynchronize());
		}
	}

	/*last iteration*/
//	printf("Last adding %lu (%d, %d) remaining %lu\n", i, blocksNum, threadsNum, blocksNum-i);
#if defined(PREFIX_SUM)
	prefixSum_before_device<<<grid, block, sharedMemSize,0>>>(total+i, (maxSize_type)blocksNum-i);
#endif
	addWithStepANDsum_device<<<grid,block,sharedMemSize,0>>>(data, size, total+i, blocksNum-i, i);
	CHK_ERROR(cudaDeviceSynchronize());

	s += data[size - 1];
	*sumOfAll = s;

//	printf("R sumOfAll = %lu \n", *sumOfAll);

}
///* Globally added mem allocators */
//__host__ void* operator new(size_t sz) throw (std::bad_alloc)
//{
////    cerr << "allocating " << sz << " bytes\n";
////    void* mem = malloc(sz);
////    if (mem)
////        return mem;
////    else
////        return NULL;
////    	//throw std::bad_alloc();
//
////    //CUDA UVA
//	cerr << "[UVA: ] allocating " << sz << " bytes\n";
//	void* mem;
//	cudaHostAlloc((void **) &mem, sz, cudaHostAllocMapped);
//
//	if (mem)
//		return mem;
//	else	{
//		cerr << "[UVA: ] error during allocation!" << endl;
//		return NULL;
//		//throw std::bad_alloc();
//	}
//
//	//throw std::bad_alloc();
//}
//
//__host__ void operator delete(void* ptr) throw()
//{
//    cerr << "deallocating at " << ptr << endl;
//    //free(ptr);
//
//	cudaFreeHost(ptr);
//
//}
