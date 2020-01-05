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
void sortByDevice(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int  blockSize)
{
    int nBins = 1 << nBits; // số bin
    int m=(n - 1) / blockSize + 1;// gridSize

    // cấp phát
    // local hist
    int **localHist = (int **)malloc(m * sizeof(int *)); 
    for (int i=0; i<m; i++)
    {
        localHist[i] = (int *)malloc(nBins * sizeof(int)); 
    }

    // scan
    int **scan = (int **)malloc(m * sizeof(int *)); 
    for (int i=0; i<m; i++)
    {
        scan[i] = (int *)malloc(nBins * sizeof(int)); 
    }

    // chỉ số bắt đầu
    int **start = (int **)malloc(m * sizeof(int *)); 
    for (int i=0; i<m; i++)
    {
        start[i] = (int *)malloc(nBins * sizeof(int)); 
    }

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;




    for (int bit = 0;  bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // gán localHist=0
        for (int i=0; i<m; i++)
        {
            memset(localHist[i], 0, nBins * sizeof(int));
        }

        // Tính localHist
        for(int blockIdx=0;blockIdx<m;blockIdx++)
        {
            for(int threadIdx=0;threadIdx<blockSize;threadIdx++)
            {
                int i=blockSize*blockIdx+threadIdx;
                if(i<n)
                {
                    int bin = (src[i] >> bit) & (nBins - 1);
                    localHist[blockIdx][bin]++;
                }
            }
        }

        // cấp phát scan=0
        for (int i=0; i<m; i++)
        {
            memset(scan[i], 0, nBins * sizeof(int));
        }

        // tính exclusive scan
        for(int bin=0; bin<nBins;bin++)
        {
            for (int blockIdx=0;blockIdx<m;blockIdx++)
            {
                if(blockIdx==0&&bin==0)
                {
                    scan[blockIdx][bin]=0;
                }
                else
                {
                    if (blockIdx==0)
                    {
                        scan[blockIdx][bin]=scan[m-1][bin-1]+localHist[m-1][bin-1];
                    }
                    else
                    {
                        scan[blockIdx][bin]=scan[blockIdx-1][bin]+localHist[blockIdx-1][bin];
                    }
                }
            }
        }

        
        // sắp xếp cục bộ
        for(int blockIdx=0;blockIdx<m;blockIdx++)
        {
            for(int index=0;index<blockSize-1;index++)
            {
                for(int threadIdx=0;threadIdx<blockSize-1-index;threadIdx++)
                {
                    int i=blockIdx*blockSize+threadIdx;
                    if(i+1<n)
                    {
                        if(((src[i] >> bit) & (nBins - 1))>((src[i+1] >> bit) & (nBins - 1)))
                        {
                            uint32_t temp=src[i];
                            src[i]=src[i+1];
                            src[i+1]=temp;
                        }
                    }
                    
                }
            }
        }

        // cấp phát start=-1
        for (int i=0; i<m; i++)
        {
            memset(start[i], -1, nBins * sizeof(int));
        }

        // tính chỉ số bắt đầu
        for(int blockIdx=0;blockIdx<m;blockIdx++)
        {
            for(int threadIdx=0;threadIdx<blockSize;threadIdx++)
            {
                int i=blockIdx*blockSize+threadIdx;
                if (i<n)
                {
                    if(threadIdx==0)
                    {
                        start[blockIdx][((src[i] >> bit) & (nBins - 1))]=threadIdx;
                    }
                    else
                    {
                        if(((src[i] >> bit) & (nBins - 1))!=((src[i-1] >> bit) & (nBins - 1)))
                        {
                            start[blockIdx][((src[i] >> bit) & (nBins - 1))]=threadIdx;
                        }
                    }
                }
            }
        }

        //scatter
        for(int blockIdx=0;blockIdx<m;blockIdx++)
        {
            for(int threadIdx=0;threadIdx<blockSize;threadIdx++)
            {
                int i=blockIdx*blockSize+threadIdx;
                if(i<n)
                {
                    int bin = (src[i] >> bit) & (nBins - 1);
                    int rank=scan[blockIdx][bin]+threadIdx-start[blockIdx][bin];
                    dst[rank]=src[i];
                }
            }
        }
        uint32_t * temp = src;
        src = dst;
        dst = temp; 
    }
    memcpy(out, src, n * sizeof(uint32_t));
    // Free memories
    for (int i=0; i<m; i++)
    {
        free(localHist[i]);
    }
    free(localHist);

    for (int i=0; i<m; i++)
    {
        free(scan[i]);
    }
    free(scan);
    for (int i=0; i<m; i++)
    {
        free(start[i]);
    }
    free(start); 
    free(originalSrc);
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
        sortByDevice(in, n, out, nBits, blockSizes);
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
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
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
    sort(in, n, out, nBits, true, blockSizes);
    // printArray(out,n);
    checkCorrectness(out, correctOut, n);
    
    // FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
