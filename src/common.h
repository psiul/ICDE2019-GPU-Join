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

#ifndef COMMON_H_
#define COMMON_H_

#include <cstdint> /*uint8_t, uint16_t, uint32_t, uint64_t*/
#include<limits.h>
#include <type_traits>

/* Constants */
#define WARP_SZ 32
#define NSTREAM 16//32
#define BDIM 1024
//Must be equal to BDIM, no?
#define SHMEMDIM 1024

#define DELIM ','

#define COMPUTE_CAPABILITY_5

#define BANKSNUM 4
#define BANKSIZE 8
#define PADSTEP BANKSNUM*BANKSIZE/sizeof(int);
#define SHIFT log2((double)BANKSNUM*BANKSIZE/sizeof(int))

__host__ __device__ __forceinline__ uint32_t hasht(uint32_t x) {
    return x;
}

#define CHUNK_SIZE ((uint64_t) (1 << 31))

constexpr uint32_t log_parts1 = 8;//9;         //< 12      2^(log_parts1 + log_parts2 + p_d + 5) ~= 'hash table size"  ~= 2 * input size
constexpr uint32_t log_parts2 = 5;//6;//8;      //< 12    

constexpr int32_t g_d        = log_parts1 + log_parts2; 
constexpr int32_t p_d        = 3;

constexpr int32_t max_chain  = (32 - 1) * 1 - 1; //(32 - 1) * 2 - 1;

#define hj_d (5 + p_d + g_d)

constexpr uint32_t hj_mask = ((1 << hj_d) - 1);

constexpr int32_t partitions = 1 << p_d; 
constexpr int32_t partitions_mask = partitions - 1;

constexpr int32_t grid_parts = 1 << g_d;
constexpr int32_t grid_parts_mask = grid_parts - 1;

constexpr uint32_t log2_bucket_size = 12;
constexpr uint32_t bucket_size      = 1 << log2_bucket_size;
constexpr uint32_t bucket_size_mask = bucket_size - 1;


#define MEM_TYPE 0

#if MEM_TYPE == 0
#define MEM_HOST
#elif MEM_TYPE == 1
#define MEM_DEVICE
#elif MEM_TYPE == 2
#define MEM_MANAGED
#elif MEM_TYPE == 3
#define MEM_S_DEVICE
#else
#define MEM_HOST
#endif

#define data_type int
#define maxSize_type unsigned long long int
#define data_min INT_MIN

extern __shared__ data_type int_shared[];
extern __shared__ maxSize_type uint64_shared[];

extern __constant__ unsigned int valuesToProcess;

extern __device__ maxSize_type sum_dev;

typedef struct timeval time_st;

typedef struct timingInfo {
	unsigned int n = 5;
	time_st start[5];
	time_st end[5];
	double greaterTime = 0;
	double reduce_usecs = 0;
	double fixPositions_usecs = 0;
	double scatter_usecs = 0;
	double copy_usecs = 0;
	double bitonic_usecs = 0;
	double total_usecs = 0;

	double greaterEventTime = 0;

	unsigned int greaterCallsNum = 0;
	unsigned int bitonicCallsNum = 0;
	unsigned int reduceCallsNum = 0;
	unsigned int fixPositionsCallsNum = 0;
} timingInfo;

union vec4{
    int4    vec ;
    int32_t i[4];
};

union vec2{
    long2   vec ;
    int64_t i[4];
};

/* Error Checking*/
#define CHK_ERROR(call)																	\
{                                                                       			\
   const cudaError_t error = call;                                      			\
   if (error != cudaSuccess)                                            			\
   {                                                                    			\
      fprintf(stderr, "GPU Error: %s:%d, ", __FILE__, __LINE__);        			\
      fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error));	\
      exit(-10*error);  															\
   }																				\
}

__device__ __forceinline__ uint32_t get_laneid(){
    uint32_t laneid;
    asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
    return laneid;
}

//__host__ void* operator new(size_t sz) throw (std::bad_alloc);
//__host__ void operator delete(void* ptr) throw();

#define USECS(start, end) (((end)->tv_sec * 1000000L + (end)->tv_usec) - ((start)->tv_sec * 1000000L + (start)->tv_usec))
#define MSECS(start, end) (((end)->tv_sec * 1000000L + (end)->tv_usec) - ((start)->tv_sec * 1000000L + (start)->tv_usec))/1000.0

void recordTime(time_st *t);

unsigned int smallestGreaterPowerOf2(const unsigned int num);
unsigned int greatestLowerPowerOf2(const unsigned int num);

void initialise_float(float *A, int N);
void initialise_int(int *A, const int N);
void printArray_int(int *A, const size_t N);
void printArray_uint(unsigned int *A, const size_t N);
void printArray_char(char *A, const maxSize_type N);
void printArray_maxSize_type(maxSize_type *A, const maxSize_type N);

void totalPrefixSum(maxSize_type *data, maxSize_type size, maxSize_type *total, maxSize_type *sumOfAll, unsigned int threadsNum, unsigned int iterNum);

/*per block*/
static __device__ void prefixSum_after(maxSize_type *data, const unsigned int size, maxSize_type *total);
__device__ void prefixSum_before(maxSize_type *data, const unsigned int size, maxSize_type *total);
__device__ void prefixSum_before(data_type *const data, const unsigned int size, data_type *total);
__device__ void prefixSum_before_multiple(data_type *const data, const unsigned int size, data_type *total, unsigned int num);
static __device__ void prefixSum_before_device(maxSize_type *data, const unsigned int size);
__device__ void prefixSum_before_multipleSeq(data_type *const data, const unsigned int sise, data_type *const borders, const unsigned int bordersNum, data_type *totalPerSeq);

static __device__ void sum(maxSize_type *data, const unsigned int size, maxSize_type *res);
static __device__ void sum_device(maxSize_type *data, const unsigned int size);

/*the whole dataset*/
__global__ void prefixSum_before(maxSize_type *data, const maxSize_type size, maxSize_type *total);
__global__ void prefixSum_before_device(maxSize_type *data, const maxSize_type size);

__global__ void sum(maxSize_type *data, const maxSize_type size, maxSize_type *res);
__global__ void sum_device(maxSize_type *data, const maxSize_type size);

__global__ void copy(data_type *dataTO, data_type *dataFROM, const maxSize_type size);
__global__ void copy(maxSize_type *dataTO, maxSize_type *dataFROM, const maxSize_type size);

__global__ void scatter(data_type *dataIN, data_type *dataOUT, maxSize_type size, maxSize_type *pos);


__device__ void prefixSum_sharedMem_before_multipleRanges(maxSize_type *data, maxSize_type data_els, maxSize_type *total1,
		maxSize_type *total2, maxSize_type *total3, unsigned int partitionsNum);


__device__ void sum(int *data, unsigned int size, int *res);

__device__ void max(int *data, unsigned int size, int *res);
__device__ void min(int *data, unsigned int size, int *res);

__global__ void aggregate(int *data, unsigned int size, int *res, int funcId);

// Handle missmatch of atomics for (u)int64/32_t with cuda's definitions
template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch_block(T *address, T val){
    return (T) atomicExch_block((unsigned int*) address, (unsigned int) val);
}


template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicExch(T *address, T val){
    return (T) atomicExch((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int),
            int>::type = 0>
__device__ __forceinline__ T atomicOr_block(T *address, T val){
    return (T) atomicOr_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicOr_block(T *address, T val){
    return (T) atomicOr_block((unsigned int*) address, (unsigned int) val);
}


template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicOr(T *address, T val){
    return (T) atomicOr((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin(T *address, T val){
    return (T) atomicMin((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicMin_block(T *address, T val){
    return (T) atomicMin_block((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd(T *address, T val){
    return (T) atomicAdd((int*) address, (int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned long long int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned long long int*) address, (unsigned long long int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(unsigned int) && std::is_integral<T>::value && !std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((unsigned int*) address, (unsigned int) val);
}

template<typename T,
         typename std::enable_if<sizeof(T) == sizeof(int) && std::is_integral<T>::value  && std::is_signed<T>::value,
            int>::type = 0>
__device__ __forceinline__ T atomicAdd_block(T *address, T val){
    return (T) atomicAdd_block((int*) address, (int) val);
}

#endif /* COMMON_H_ */
