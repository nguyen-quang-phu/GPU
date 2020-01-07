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

__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums)
{   
    extern __shared__ uint32_t value[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        value[threadIdx.x] = in[i];
    }
    
    for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
        __syncthreads();
        int tmp;
        if (threadIdx.x < n - stride)
            tmp = value[threadIdx.x-stride];
        else
            tmp = 0;
        __syncthreads();
        value[threadIdx.x] += tmp;
    }
    
    blkSums[blockIdx.x] = value[blockDim.x - 1];
    if (i<n) {
        out[i]=value[threadIdx.x];
    }
}
__global__ void radixSort1bit(uint32_t * in, int n, uint32_t * out,int nBits, int bit,int nBins)
{   int i = blockIdx.x * blockDim.x + threadIdx.x;
    for(int indexbit=0;indexbit<nBits;indexbit++)
    {
        extern __shared__ uint32_t value[];
        
        if (i < n) 
        {
            value[threadIdx.x] = ((((in[i] >> bit) & (nBins - 1)) >> indexbit) & 1);
        }
        __syncthreads();
        for(int stride=1;stride<blockDim.x;stride*=2)
        {
            int temp=0;
            if(threadIdx.x>=stride)
            {
                temp=value[threadIdx.x-stride];// lấy phần tử trước đó stride bước
            }
            __syncthreads();// chắc chắn giá trị năm trước stride bước đã được lấy vào bộ nhớ thanh ghi
            if(threadIdx.x>=stride )
            {
                value[threadIdx.x]+=temp;
            }
            __syncthreads();// chắc chắn các giá trị đã được cộng xong
        }
        int nZeros=0;
        if(blockIdx.x*blockDim.x+blockDim.x<=n)
            nZeros = blockDim.x - value[blockDim.x-2] -((((in[blockIdx.x*blockDim.x+blockDim.x-1] >> bit) & (nBins - 1)) >> indexbit) & 1);
        else
        {
            if(n%blockDim.x>=2)
            nZeros = n%blockDim.x - value[n%blockDim.x-2] - ((((in[n-1] >> bit) & (nBins - 1)) >> indexbit) & 1);
            else
            nZeros = n%blockDim.x  - ((((in[n-1] >> bit) & (nBins - 1)) >> indexbit) & 1);
        }
        if (i<n)
        {
            if(threadIdx.x==0)
            {
                if (((((in[i] >> bit) & (nBins - 1)) >> indexbit) & 1)==0)
                {
                    out[i]=in[i];
                }
                else
                    out[nZeros+blockIdx.x*blockDim.x]=in[i];
            }
            else
            {
                if(((((in[i] >> bit) & (nBins - 1)) >> indexbit) & 1)==0)
                {
                    out[i-value[threadIdx.x-1]]=in[i];
                }
                else
                {
                    out[nZeros+value[threadIdx.x-1]+blockIdx.x*blockDim.x]=in[i];
                }
            }
        }
        __syncthreads();
            uint32_t *tam=in;
            in=out;
            out=tam;
    }
    
}

__global__ void addSumScan(int * out, int n, int * blkSums)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0) 
    {
        out[i] = out[i] + blkSums[blockIdx.x - 1];
    }
}



void sortByDevice(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int  blockSize)
{
    int nBins = 1 << nBits; // số bin
    int m=(n - 1) / blockSize + 1;// gridSize
    dim3 blockSizeHist(blockSize);
    dim3 blockSizeScan(blockSize);

    dim3 gridSizeHist((n - 1) / blockSizeHist.x + 1);
    dim3 gridSizeScan((nBins*m - 1) / blockSizeScan.x + 1);
    // cấp phát
   
    // scan
    int *d_scan, *d_blkSums, *d_histScan, *d_blkOuts;
    int *histScan = (int *)malloc(m*nBins * sizeof(int));
    int *blkSums = (int *)malloc(m*nBins*sizeof(int));

    CHECK(cudaMalloc(&d_scan, nBins*m * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums,gridSizeScan.x*sizeof(int)));
    CHECK(cudaMalloc(&d_blkOuts,m*nBins*sizeof(int)));

    CHECK(cudaMalloc(&d_histScan,m*nBins*sizeof(int)));
    // chỉ số bắt đầu
    int **start = (int **)malloc(m * sizeof(int *)); 
    for (int i=0; i<m; i++)
    {
        start[i] = (int *)malloc(nBins * sizeof(int)); 
    }

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * d_in,*d_out;
    CHECK(cudaMalloc(&d_in,n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_out,n * sizeof(uint32_t)));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;
    // CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));

    for (int bit = 0;  bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        CHECK(cudaMemcpy(d_in, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        // Tính local hist bỏ vào d_scan
        computeLocalHist<<<gridSizeHist, blockSizeHist, blockSize*sizeof(int)>>>(d_in, n, d_scan, nBins,bit);
       
        // // Tính exclusive scan bỏ vào d_histscan
        scanBlkKernel<<<gridSizeScan,blockSizeScan,blockSize*sizeof(int)>>>(d_scan,m*nBins,d_histScan,d_blkSums);
        size_t bytes = gridSizeScan.x * sizeof(int);
        int * in_tmp = (int *)malloc(bytes);
        int * out_tmp = (int *)malloc(bytes);

        CHECK(cudaMemcpy(in_tmp, d_blkSums, gridSizeScan.x * sizeof(int), cudaMemcpyDeviceToHost));
        out_tmp[0] = in_tmp[0];
	    for (int i = 1; i < gridSizeScan.x; i++)
	    {
	    	out_tmp[i] = out_tmp[i - 1] + in_tmp[i];
	    }

		CHECK(cudaMemcpy(d_blkOuts, out_tmp, gridSizeScan.x * sizeof(int), cudaMemcpyHostToDevice));
        addSumScan<<<gridSizeScan,blockSizeScan>>>(d_histScan, n, d_blkOuts);

    	cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        histScan[0] = 0;
		CHECK(cudaMemcpy(histScan + 1, d_histScan, (nBins*m - 1) * sizeof(int), cudaMemcpyDeviceToHost));
        
        
        // sắp xếp cục bộ
        // for(int blockIdx=0;blockIdx<m;blockIdx++)
        // {
        //     for(int index=0;index<blockSize-1;index++)
        //     {
        //         for(int threadIdx=0;threadIdx<blockSize-1-index;threadIdx++)
        //         {
        //             int i=blockIdx*blockSize+threadIdx;
        //             if(i+1<n)
        //             {
        //                 if(((src[i] >> bit) & (nBins - 1))>((src[i+1] >> bit) & (nBins - 1)))
        //                 {
        //                     uint32_t temp=src[i];
        //                     src[i]=src[i+1];
        //                     src[i+1]=temp;
        //                 }
        //             }
                    
        //         }
        //     }
        // }

        radixSort1bit<<<gridSizeHist,blockSizeHist,blockSize*sizeof(uint32_t)>>>(d_in,n,d_out,nBits,bit,nBins);
        CHECK(cudaMemcpy(src,d_in,n*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        for(int i=0;i<50;i++)
        printf("%d ",src[i]);
        printf("\n\n\n");

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
    // cudaFree(d_blkSums);
    cudaFree(d_histScan);
    cudaFree(d_in);
    cudaFree(d_out);
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
    // n = 600000;
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