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


// Thrust Radix Sort
void sortByThrust(const uint32_t * in, int n, 
    uint32_t * out)
{
    thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}


__global__ void computeLocalHist(uint32_t * in, int n, uint32_t * scan, int nBins, int bit)
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


__global__ void scanBlkKernel(uint32_t * in, int n, uint32_t * out, uint32_t * blkSums)
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
    __syncthreads();
    if (i<n) {
        if(threadIdx.x==0)
        {
            out[i]=0;
        } 
        else
        out[i]=value[threadIdx.x-1];
    }
}



__global__ void addSumScan(uint32_t * out, int n, uint32_t * blkSums)
{   
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0) 
    {
        out[i] = out[i] + blkSums[blockIdx.x - 1];
    }
}



__global__ void radixSort1bit(uint32_t * in, int n, uint32_t * out,int nBits, int bit,int nBins, uint32_t* starts)
{   int i = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ uint32_t value[];
    __shared__ uint32_t start[256];

    for(int indexbit=0;indexbit<nBits;indexbit++)
    {
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
        uint32_t *tmp=in;
        in=out;
        out=tmp;
    }
    if (i<n)
    {
        if(threadIdx.x==0)
        {
            start[((in[i] >> bit) & (nBins - 1))]=threadIdx.x;
        }
        else
        {
            if(((in[i] >> bit) & (nBins - 1))!=((in[i-1] >> bit) & (nBins - 1)))
            {
                start[((in[i] >> bit) & (nBins - 1))]=threadIdx.x;
                starts[blockIdx.x*nBins+((in[i] >> bit) & (nBins - 1))]=start[((in[i] >> bit) & (nBins - 1))];
            }
        }
    }    
}


__global__ void scatter(uint32_t * in, int n, uint32_t * out,int nBits, int bit,int nBins, uint32_t* start, uint32_t* histScan)
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
    {
        int bin = (in[i] >> bit) & (nBins - 1);
        int rank=histScan[bin*gridDim.x+blockIdx.x]+threadIdx.x-start[nBins*blockIdx.x+bin];
        out[rank]=in[i];
    }
}


void sortByDevice(const uint32_t * in, int n, 
    uint32_t * out, 
    int nBits, int  blockSizes)
{
    int nBins = 1 << nBits; // số bin
    int m = (n - 1) / blockSizes + 1;// gridSize
    dim3 blockSize(blockSizes);
    dim3 blockSizeScan(blockSizes);

    dim3 gridSize((n - 1) / blockSize.x + 1);
    dim3 gridSizeScan((nBins*m - 1) / blockSizeScan.x + 1);
    // cấp phát
   
    // scan
    uint32_t *d_scan, *d_blkSums, *d_histScan, *d_blkOuts, *d_starts;
    uint32_t *histScan = (uint32_t *)malloc(m*nBins * sizeof(uint32_t));
    uint32_t *blkSums = (uint32_t *)malloc(m*nBins*sizeof(uint32_t));
    uint32_t* starts1D=(uint32_t *) malloc(m*nBins*sizeof(uint32_t));

    CHECK(cudaMalloc(&d_scan, nBins*m * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_blkSums,gridSizeScan.x*sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_blkOuts,m*nBins*sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_starts,m*nBins*sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_histScan,m*nBins*sizeof(uint32_t)));
    // chỉ số bắt đầu

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * d_in,*d_out, *d_tmp;
    CHECK(cudaMalloc(&d_in,n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_out,n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_tmp,n * sizeof(uint32_t)));

    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    size_t bytes = gridSizeScan.x * sizeof(uint32_t);
    uint32_t * in_tmp = (uint32_t *)malloc(bytes);
    uint32_t * out_tmp = (uint32_t*)malloc(bytes);
    CHECK(cudaMemcpy(d_in, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    GpuTimer timerTmp1,timerTmp2,timerTmp3,timerTmp4,timerTmp5; 
    float time1,time2,time3,time4,time5;
    time1=time2=time3=time4=time5=0;


    for (int bit = 0;  bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // Tính local hist bỏ vào d_scan
        timerTmp1.Start();
        computeLocalHist<<<gridSize, blockSize, blockSizes*sizeof(uint32_t)>>>(d_in, n, d_scan, nBins,bit);
        timerTmp1.Stop();
        time1 = time1 + timerTmp1.Elapsed();
        timerTmp2.Start(); 

        // // Tính exclusive scan bỏ vào d_histscan
        scanBlkKernel<<<gridSizeScan,blockSizeScan,blockSizes*sizeof(uint32_t)>>>(d_scan,m*nBins,d_histScan,d_blkSums);
        CHECK(cudaMemcpy(in_tmp, d_blkSums, gridSizeScan.x * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        out_tmp[0] = in_tmp[0];
	    for (int i = 1; i < gridSizeScan.x; i++)
	    {
	    	out_tmp[i] = out_tmp[i - 1] + in_tmp[i];
	    }
		CHECK(cudaMemcpy(d_blkOuts, out_tmp, gridSizeScan.x * sizeof(uint32_t), cudaMemcpyHostToDevice));
        addSumScan<<<gridSizeScan,blockSizeScan>>>(d_histScan, n, d_blkOuts);
    	cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        timerTmp2.Stop();
        time2 = time2 + timerTmp2.Elapsed();
        timerTmp3.Start();

        // Radix Sort 1 bit
        radixSort1bit<<<gridSize,blockSize,blockSizes*sizeof(uint32_t)>>>(d_in,n,d_out,nBits,bit,nBins, d_starts);

        timerTmp3.Stop();
        time3 = time3 + timerTmp3.Elapsed();
        timerTmp5.Start();

        // Scatter
        scatter<<<gridSize,blockSize,blockSizes*sizeof(uint32_t)>>>(d_in,n,d_out,nBits,bit,nBins,d_starts,d_histScan);

        timerTmp5.Stop();
        time5 = time5 + timerTmp5.Elapsed();

        d_tmp = d_in;
        d_in = d_out;
        d_out = d_tmp;
    }
    printf("Time (local hist): %.3f ms\n", time1);
    printf("Time (exclusive scan): %.3f ms\n", time2);
    printf("Time (local sort) + Time (start value): %.3f ms\n", time3);
    printf("Time (scatter): %.3f ms\n", time5);

    
    CHECK(cudaMemcpy(src, d_in, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    memcpy(out, src, n * sizeof(uint32_t));
    // Free memories
    cudaFree(d_scan);
    cudaFree(d_blkSums);
    cudaFree(d_histScan);
    cudaFree(d_in);
    cudaFree(d_out);
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
            // printf("%i %i",out[i+100] , correctOut[i+100]);
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < 100; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    // n = 1000000;
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
    printf("\nblock size: %d", blockSizes);

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