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

__global__ void computeLocalHist(uint32_t * in, int n, int * scan, int nBins, int bit)
{
    extern __shared__ int s_hist[];
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    for(int stride=0;stride<nBins;stride+=blockDim.x)
        if(threadIdx.x+stride<nBins)
            s_hist[threadIdx.x+stride]=0;
    __syncthreads();
    if(i<n)
    {
        int bin=(in[i]>>bit)&(nBins-1);// lấy nBits ra để tính xem phần tử này thuộc bin nào
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();// syncthreads để chắc chắn các phần tử trong block đã được tính trong s_hist

    for(int stride=0;stride<nBins;stride+=blockDim.x)
        if(threadIdx.x+stride<nBins)
            scan[(threadIdx.x+stride)*gridDim.x+blockIdx.x]=s_hist[threadIdx.x+stride];
            // hist[nBins*blockIdx.x+threadIdx.x+stride]=s_hist[threadIdx.x+stride];
}
__global__ void addBlkKernel(int * in, int n, int * blkSums)
{
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    if(i<n&&blockIdx.x>0)
        in[i]+=blkSums[blockIdx.x-1];
}
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums)
{   
    // TODO
    extern __shared__ int s_in[];
    int i=blockDim.x*blockIdx.x+threadIdx.x;
    
    // gán giá trị tương ứng vào smem
    if(i<n)
        s_in[threadIdx.x]=in[i];
    else
        s_in[threadIdx.x]=0;
    __syncthreads();

    // cộng các giá cách nhau stride bước lại với nhau
    for(int stride=1;stride<blockDim.x;stride*=2)
    {
        int temp=0;
        if(threadIdx.x>=stride)
        {
            temp=s_in[threadIdx.x-stride];// lấy phần tử trước đó stride bước
        }
        __syncthreads();// chắc chắn giá trị năm trước stride bước đã được lấy vào bộ nhớ thanh ghi
        if(threadIdx.x>=stride )
        {
            s_in[threadIdx.x]+=temp;
        }
        __syncthreads();// chắc chắn các giá trị đã được cộng xong
    }

    // gán giá trị tương ứng vào mảng out
    if(i<n)
        out[i]=s_in[threadIdx.x];

    // thread cuối cùng trong block ghi giá trị vào blkSums theo blockIdx
    if(blkSums!=NULL)
    {
        if(threadIdx.x==blockDim.x-1)
        {
            blkSums[blockIdx.x]=s_in[threadIdx.x];
        }
    }
}
void sortByDevice(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int  blockSize)
{
    int nBins = 1 << nBits; // số bin
    int m=(n - 1) / blockSize + 1;// gridSize
    dim3 blockSize1(blockSize);
    dim3 gridSize1((n - 1) / blockSize1.x + 1);
    dim3 gridSize2((nBins*m - 1) / blockSize + 1);
    // cấp phát
   
    // scan
    int *d_scan,*d_blkSums,*d_histScan;
    int * histScan = (int *)malloc(m*nBins * sizeof(int));
    int* blkSums = (int *)malloc(gridSize2.x*sizeof(int));

    CHECK(cudaMalloc(&d_scan, nBins*m * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums,gridSize2.x*sizeof(int)));
    CHECK(cudaMalloc(&d_histScan,m*nBins*sizeof(int)));
    // chỉ số bắt đầu
    int **start = (int **)malloc(m * sizeof(int *)); 
    for (int i=0; i<m; i++)
    {
        start[i] = (int *)malloc(nBins * sizeof(int)); 
    }

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * d_in;
    CHECK(cudaMalloc(&d_in,n * sizeof(uint32_t)));

    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    for (int bit = 0;  bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));
        // Tính local hist bỏ vào d_scan
        computeLocalHist<<<gridSize1, blockSize1, blockSize*sizeof(int)>>>(d_in, n, d_scan, nBins,bit);
        // // Tính exclusive scan bỏ vào d_histscan
        // scanBlkKernel<<<gridSize2,blockSize1,blockSize*sizeof(int)>>>(d_scan,m*nBins,d_histScan,d_blkSums);
        // CHECK(cudaMemcpy(histScan,d_histScan,nBins*m*sizeof(int),cudaMemcpyDeviceToHost));
        // CHECK(cudaMemcpy(blkSums,d_blkSums,gridSize2.x*sizeof(int),cudaMemcpyDeviceToHost));
        // for(int i=1;i<gridSize1.x;i++)
        // {
        //     blkSums[i]+=blkSums[i-1];
        // }
        // CHECK(cudaMemcpy(d_blkSums,blkSums,gridSize2.x*sizeof(int),cudaMemcpyHostToDevice));
        // addBlkKernel<<<gridSize1,blockSize1>>>(d_histScan,nBins,d_blkSums);
        // CHECK(cudaMemcpy(&histScan[1],d_histScan,(m*nBins-1)*sizeof(int),cudaMemcpyDeviceToHost));

        uint32_t *scan=(uint32_t*) malloc(nBins*m*sizeof(uint32_t));
        CHECK(cudaMemcpy(scan,d_scan,nBins*m*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        histScan[0]=0;
        for(int i=1;i<nBins*m;i++)
            histScan[i]=histScan[i-1]+scan[i-1];

        
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
                    int rank=histScan[bin*m+blockIdx]+threadIdx-start[blockIdx][bin];
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
    cudaFree(d_scan);
    cudaFree(d_blkSums);
    cudaFree(d_histScan);
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
    n = 600000;
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