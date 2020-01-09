#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <algorithm> 
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

// Sequential radix sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
void sortByThrust(const uint32_t * in, int n, 
    uint32_t * out)
{
    thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}


#define getDigit(val, start, radix) ((val>>start) & (radix - 1))
#define ITEMSPERTHREAD 4

// basically memset, but for unsigned int's only
__device__ void k_assignRange(unsigned int * start, unsigned int value, unsigned int numElems)
{
    for (int i = 0; i < numElems; i++)
      *(start + i) = value;    
}

// basically memcpy, but for unsigned int's only
__device__ void k_copyRange(unsigned int * output, unsigned int * input, unsigned int numElems)
{
    for (int i = 0; i < numElems; i++)
      output[i] = input[i];    
}


// performs a prefix sum, where each thread has the value pred
__device__ unsigned int scan(unsigned int pred, unsigned int * sh_sums)
{
    sh_sums[threadIdx.x] = pred;
    __syncthreads();
    
    for (int skip = 1; skip < blockDim.x; skip *= 2)
    {
        int newValue = (threadIdx.x >= skip) ? sh_sums[threadIdx.x] + sh_sums[threadIdx.x - skip] : sh_sums[threadIdx.x];
        __syncthreads();
        sh_sums[threadIdx.x] = newValue;
        __syncthreads();
    }    
    if (threadIdx.x > 0)
        return sh_sums[threadIdx.x - 1];
    else
        return 0;
}



// like above, where each thread handles multiple values and outputs several different values in the prefix sum
// itemsThisThread will equal itesmPerThread except if this block has threads that have no items to work with
__device__ void scanMultiple(unsigned int * outputSums, 
                               unsigned int * inputVals,
                               unsigned int localID, // first value in sh_sums[] this thread deals with
                               unsigned int numElems,  
                               unsigned int * sh_sums, // shared memory for computing the sums
                               unsigned int itemsThisThread) // # of items this thread works with
{
    k_copyRange(sh_sums + localID, inputVals, itemsThisThread);
    __syncthreads();
    
    unsigned int newValues[ITEMSPERTHREAD];

    for (int skip = 1; skip < numElems; skip *= 2) 
    {
        for (int i = 0; i < itemsThisThread; i++)
        {
            if (localID + i >= skip)
                newValues[i] = sh_sums[localID + i] + sh_sums[localID + i - skip];
            else
                newValues[i] = sh_sums[localID + i];
        }
        __syncthreads();
        k_copyRange(sh_sums + localID, newValues, itemsThisThread); 
        __syncthreads();  
    }
    
    // write output
    if (threadIdx.x > 0)
        outputSums[0] = sh_sums[localID - 1];
    else
        outputSums[0] = 0;  
    k_copyRange(outputSums + 1, sh_sums + localID, itemsThisThread - 1);  
}


// outputs the "rank" of each item, for partitioning the block by predicate value
__device__ unsigned int split(bool pred, unsigned int blocksize, unsigned int * sh_sums)
{
    unsigned int true_before = scan(pred, sh_sums);
    __shared__ unsigned int false_total;    
    if(threadIdx.x == blocksize - 1)
        false_total = blocksize - (true_before + pred);
    __syncthreads();  
    if(pred) 
        return true_before + false_total;
    else 
        return threadIdx.x - true_before; 
}

// single-block radix sort
__global__ void k_radixSortBlock(unsigned int * d_vals,
                                 unsigned int startBit,
                                 unsigned int radix,
                                 unsigned int numElems)
{
    int inputID = threadIdx.x + blockDim.x * blockIdx.x;
    if (inputID < numElems)
    {
		extern __shared__ unsigned int sh_arr[];
		unsigned int * sh_sums = sh_arr;
		unsigned int * sh_vals = sh_arr + blockDim.x;
		sh_vals[threadIdx.x] = d_vals[inputID];
		__syncthreads();
		
		for (int d = 1; d < radix; d <<= 1)
		{
			unsigned int i = split(((sh_vals[threadIdx.x]>>startBit) & d) > 0, min(numElems - blockDim.x * blockIdx.x, blockDim.x), sh_sums);
			unsigned int oldValue = sh_vals[threadIdx.x];
			__syncthreads();
			sh_vals[i] = oldValue;
			__syncthreads();
		}  
		d_vals[blockDim.x * blockIdx.x + threadIdx.x] = sh_vals[threadIdx.x];
	}
}




// 1st step to calculating  globalOffsets (offsets for each block and each radix)
    // this step simply counts the # of each radix value in each block, later we do a scan on that to get globalOffsets
// as well as calculating localOffsets (offsets within each block for each radix)
// d_bucketSize[i][j] is the # of elements of radix i that are in block j
// send only enough threads to look at blockSize - 1 items (since each thread compares to the next item in the block)
__global__ void k_findOffsets(unsigned int * d_globalOffsets,
                              unsigned int * d_localOffsets,
                              unsigned int * d_vals,
                              unsigned int startBit,
                              unsigned int radix,
                              unsigned int numElems)
{

    extern __shared__ unsigned int sh_arr[]; // for storing offsets
    unsigned int inputID = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int blockSize = min(numElems - blockDim.x * blockIdx.x, blockDim.x);
    int thisDigit = getDigit(d_vals[inputID], startBit, radix);
    
    if (inputID < numElems)
	{
		// missing radix values before the first item?  1st thread gives their offset as 0
		if (threadIdx.x == 0)
			k_assignRange(sh_arr, 0, thisDigit + 1);
		
		// missing radix values after the last item?  last thread gives their offset as blockSize
		if (threadIdx.x == blockSize - 1)
			k_assignRange(sh_arr + thisDigit + 1, blockSize, radix - 1 - thisDigit);
		else
		{
			int nextDigit = getDigit(d_vals[inputID + 1], startBit, radix);    		
			// assign offsets for all value(s) between this digit and the next digit, including the next digit but not this one 
			if (nextDigit > thisDigit)
				k_assignRange(sh_arr + thisDigit + 1, threadIdx.x + 1, nextDigit - thisDigit);
		}

		__syncthreads();
		
		// index both output arrays in bucket-major order
		unsigned int outputID = blockIdx.x + gridDim.x  * threadIdx.x;
		
		if (threadIdx.x < radix - 1)
		{
			d_localOffsets[outputID] = sh_arr[threadIdx.x];
			d_globalOffsets[outputID] = sh_arr[threadIdx.x + 1] - sh_arr[threadIdx.x];
		}
		else if (threadIdx.x == radix - 1)
		{
			d_localOffsets[outputID] = sh_arr[threadIdx.x];
			d_globalOffsets[outputID] = blockSize - sh_arr[threadIdx.x];
		}   
	}
}

__global__ void k_scan(unsigned int * d_vals,
                               unsigned int numElems)
{
    unsigned int localID = threadIdx.x * ITEMSPERTHREAD;
    unsigned int inputID = blockDim.x * blockIdx.x + localID;
    unsigned int itemsThisThread = min(numElems - localID, ITEMSPERTHREAD);   
    if (inputID >= numElems)
        return;   
    extern __shared__ unsigned int sh_arr[];
    unsigned int outputSums[ITEMSPERTHREAD];   
    scanMultiple(outputSums, d_vals + inputID, localID, numElems, sh_arr, itemsThisThread);
    __syncthreads();   
    k_copyRange(d_vals + inputID, outputSums, itemsThisThread);   
}



__global__ void k_scatter(unsigned int * d_outputVals, 
                          unsigned int * d_inputVals,
                          unsigned int * d_localOffsets, 
                          unsigned int * d_globalOffsets, 
                          int startBit,
                          int radix,
                          unsigned int numElems)
{
    unsigned int inputID = blockDim.x * blockIdx.x + threadIdx.x;
    if (inputID >= numElems)
        return;
    
    int thisDigit = getDigit(d_inputVals[inputID], startBit, radix);
    unsigned int offsetIndex = gridDim.x * thisDigit + blockIdx.x;
    unsigned int outputID = threadIdx.x - d_localOffsets[offsetIndex] + d_globalOffsets[offsetIndex];
    d_outputVals[outputID] = d_inputVals[inputID];
}

    
// swap pointers
void exch(unsigned int * * a, unsigned int *  * b)
{
    unsigned int * temp = *a;
    *a = *b;
    *b = temp;    
}



// based off of this paper: http://mgarland.org/files/papers/nvr-2008-001.pdf
void radix_sort (unsigned int* const d_inputVals,
               unsigned int* const d_outputVals,
               const size_t numElems)
{
    
    // PREFERENCES
    const unsigned int blockSize = 512;   
    const int numBits = 3;
    const unsigned int radix = pow(2,numBits); 
    unsigned int numBlocks = (numElems - 1) / blockSize + 1;
    assert((radix * numBlocks - 1) / ITEMSPERTHREAD + 1 < 1024);
    
    unsigned int * d_globalOffsets, * d_localOffsets;
    CHECK(cudaMalloc(&d_globalOffsets, radix * numBlocks * sizeof(unsigned int)));
    CHECK(cudaMalloc(&d_localOffsets, radix * numBlocks * sizeof(unsigned int)));
    
    unsigned int * d_valuesA = d_inputVals; 
    unsigned int * d_valuesB = d_outputVals;
    
    for (int d = 0; d < 32; d += numBits)
    {
        k_radixSortBlock<<<numBlocks, blockSize, 3 * blockSize*sizeof(unsigned int)>>>(d_valuesA, d, radix, numElems);
        k_findOffsets<<<numBlocks, blockSize, (radix + 1)*sizeof(unsigned int)>>>(d_globalOffsets, d_localOffsets, d_valuesA, d, radix, numElems);
        k_scan<<<1, (radix * numBlocks - 1) / ITEMSPERTHREAD + 1, radix * numBlocks * sizeof(unsigned int)>>>(d_globalOffsets,numElems);
        k_scatter<<<numBlocks, blockSize>>>(d_valuesB, d_valuesA, d_localOffsets, d_globalOffsets, d, radix, numElems); 
        exch(&d_valuesA, &d_valuesB);
    }
}





// Radix sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        bool useDevice=false, int  blockSizes=512)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix sort by Thrust\n");
        sortByThrust(in, n, out);
    }
    else // use device
    {
    	printf("\nRadix sort by device\n");

        unsigned int* d_in;
        unsigned int* d_out;
        CHECK(cudaMalloc(&d_in, sizeof(unsigned int) * n));
        CHECK(cudaMalloc(&d_out, sizeof(unsigned int) * n));
        CHECK(cudaMemcpy(d_in, in, sizeof(unsigned int) * n, cudaMemcpyHostToDevice));
        radix_sort(d_in, d_out, n);
        CHECK(cudaMemcpy(out, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost));
        CHECK(cudaFree(d_out));
        CHECK(cudaFree(d_in));

    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n");
}

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}




int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    n = 65536;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    // unsigned int * in1 = (unsigned int *)malloc(n*sizeof(unsigned int));

    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    // unsigned int * out1 = (unsigned int *)malloc(n*sizeof(unsigned int));

    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    //printArray(in, n);

    // SET UP NBITS
    int nBits = 4; // Default
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // DETERMINE BLOCK SIZES
    int blockSizes=512; // One for histogram, one for scan
    if (argc == 3)
    {
        blockSizes = atoi(argv[2]);
    }
    printf("\block size: %d", blockSizes);

    // SORT BY HOST
    sort(in, n, correctOut, nBits);
    // printArray(correctOut, n);
    
    // SORT BY DEVICE

    // GpuTimer timer; 
    // timer.Start();

    // unsigned int* d_in;
    // unsigned int* d_out;
    // CHECK(cudaMalloc(&d_in, sizeof(unsigned int) * n));
    // CHECK(cudaMalloc(&d_out, sizeof(unsigned int) * n));
    // CHECK(cudaMemcpy(d_in, in, sizeof(unsigned int) * n, cudaMemcpyHostToDevice));
    // radix_sort(d_in, d_out, n);
    // CHECK(cudaMemcpy(out, d_out, sizeof(unsigned int) * n, cudaMemcpyDeviceToHost));
    // CHECK(cudaFree(d_out));
    // CHECK(cudaFree(d_in));

    // timer.Stop();
    // printf("Time: %.3f ms\n", timer.Elapsed());



    sort(in, n, out, nBits, true, blockSizes);
    // printArray(out,n);
    checkCorrectness(out, correctOut, n);
    
    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
