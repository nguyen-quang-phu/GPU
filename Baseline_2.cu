#include <stdio.h>
#include <stdint.h>

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
void sortByHost(const uint32_t * in, int n,
                uint32_t * out,
                int nBits)
{
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // TODO: Compute "hist" of the current digit
        memset(hist,0,nBins*sizeof(int));
       
        for(int i=0;i<n;i++)
        {
            int bin=(src[i]>>bit)&(nBins-1);
            hist[bin]++;
        }
    	// TODO: Scan "hist" (exclusively) and save the result to "histScan"
        histScan[0]=0;
        for(int bin=1;bin<nBins;bin++)
        {
            histScan[bin]=histScan[bin-1]+hist[bin-1];
        }
    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
        for(int i=0;i<n;i++)
        {
            int bin=(src[i]>>bit)&(nBins-1);
            dst[histScan[bin]]=src[i];
            histScan[bin]++;

        }
        // TODO: Swap "src" and "dst"
        uint32_t * tmp=src;
        src=dst;
        dst=tmp;

    }

    // TODO: Copy result to "out"
    memcpy(out,src,n * sizeof(uint32_t));
    // Free memories
    free(hist);
    free(histScan);
    free(originalSrc);
}
/*
Use SMEM.
*/
__global__ void computeHistKernel2(int * in, int n, int * hist, int nBins, int bit)
{
    // TODO
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_hist[];
    int i=blockDim.x*blockIdx.x+threadIdx.x;

    // Gán giá trị 0 cho local hist
    // Nếu nBins> blockDim.x thì mỗi thread sẽ gán giá trị cho phần tử bin cách mỗi stride=blockDim.x
    for(int stride=0;stride<nBins;stride+=blockDim.x)
        if(threadIdx.x+stride<nBins)
            s_hist[threadIdx.x+stride]=0;
    __syncthreads();// syncthreads để chắc chắn các phần tử trong s_hist đã được gắn giá trị 0

    // Tính local hist
    if(i<n)
    {
        int bin=(in[i]>>bit)&(nBins-1);// lấy nBits ra để tính xem phần tử này thuộc bin nào
        atomicAdd(&s_hist[bin], 1);
    }
    __syncthreads();// syncthreads để chắc chắn các phần tử trong block đã được tính trong s_hist

    // Each block adds its local hist to global hist using atomic on GMEM
    for(int stride=0;stride<nBins;stride+=blockDim.x)
        if(threadIdx.x+stride<nBins)
            atomicAdd(&hist[threadIdx.x+stride],s_hist[threadIdx.x+stride]);
}
// TODO: You can define necessary functions here
// Cộng giá trị blkSum vào các phần tử tương ứng
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
// (Partially) Parallel radix sort: implement parallel histogram and parallel scan in counting sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
// Why "int * blockSizes"? 
// Because we may want different block sizes for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{
    // TODO
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
    // (using STABLE counting sort)

    // Khởi tạo các thông số blockSize,gridSize cho hist và histScan
    dim3 blockSize1(blockSizes[0]);
    dim3 gridSize1((n - 1) / blockSize1.x + 1);
    dim3 blockSize2(blockSizes[1]);
    dim3 gridSize2((n - 1) / blockSize2.x + 1);

    // Allocate device memories
    int * d_in, * d_hist,*d_histScan,*d_blkSums;
    CHECK(cudaMalloc(&d_in, n * sizeof(int)));
    CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));
    CHECK(cudaMalloc(&d_histScan,nBins*sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums,gridSize2.x*sizeof(int)));

    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // TODO: Compute "hist" of the current digit
        CHECK(cudaMemset(d_hist,0,nBins*sizeof(int)));
        CHECK(cudaMemcpy(d_in, src, n * sizeof(int), cudaMemcpyHostToDevice));

        // tính hist
        computeHistKernel2<<<gridSize1, blockSize1, nBins*sizeof(int)>>>(d_in, n, d_hist, nBins,bit);

        // bắt lỗi hàm kernel
        cudaError_t errSync  = cudaGetLastError();
        cudaError_t errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
        {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
            return;
        }
            
        if (errAsync != cudaSuccess)
        {
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
            return;
        }

        // TODO: Scan "hist" (exclusively) and save the result to "histScan"

        int* blkSums = (int *)malloc(gridSize2.x*sizeof(int));
        
        scanBlkKernel<<<gridSize2,blockSize2,blockSize2.x*sizeof(int)>>>(d_hist,nBins,d_histScan,d_blkSums);

        // bắt lỗi hàm kernel
        errSync  = cudaGetLastError();
        errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
        {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
            return;
        }
            
        if (errAsync != cudaSuccess)
        {
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
            return;
        }
        CHECK(cudaMemcpy(blkSums,d_blkSums,gridSize2.x*sizeof(int),cudaMemcpyDeviceToHost));
        for(int i=1;i<gridSize2.x;i++)
        {
            blkSums[i]+=blkSums[i-1];
        }
        CHECK(cudaMemcpy(d_blkSums,blkSums,gridSize2.x*sizeof(int),cudaMemcpyHostToDevice));
        addBlkKernel<<<gridSize2,blockSize2>>>(d_histScan,nBins,d_blkSums);
        // bắt lỗi hàm kernel
        errSync  = cudaGetLastError();
        errAsync = cudaDeviceSynchronize();
        if (errSync != cudaSuccess) 
        {
            printf("Sync kernel error: %s\n", cudaGetErrorString(errSync));
            return;
        }
            
        if (errAsync != cudaSuccess)
        {
            printf("Async kernel error: %s\n", cudaGetErrorString(errAsync));
            return;
        }

        // copy lệch qua 1 phần tử
        CHECK(cudaMemcpy(&histScan[1],d_histScan,(nBins-1)*sizeof(int),cudaMemcpyDeviceToHost));
        histScan[0]=0;
    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
        for(int i=0;i<n;i++)
        {
            int bin=(src[i]>>bit)&(nBins-1);
            dst[histScan[bin]]=src[i];
            histScan[bin]++;

        }
        // TODO: Swap "src" and "dst"
        uint32_t * tmp=src;
        src=dst;
        dst=tmp;

    }

    // TODO: Copy result to "out"
    memcpy(out,src,n * sizeof(uint32_t));
    // Free memories

    free(hist);
    free(histScan);
    free(originalSrc);
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_histScan));
    CHECK(cudaFree(d_blkSums));
}

// Radix sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        bool useDevice=false, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits);
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
    n = 1000000;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    // printArray(in, n);

    // SET UP NBITS
    int nBits = 4; // Default
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

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
