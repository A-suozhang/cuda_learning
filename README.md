# [CUDA Learning]

Date Created: July 12, 2024 9:40 AM
Status: 研究

# CUDA Intro-tutorial

> 主要参考了PaddleJITLab的Intro材料：[CUDATutorial | Notebook (keter.top)](https://cuda.keter.top/)，作者还有一些其他的Blog: [Aurelius84 - 博客园 (cnblogs.com)](https://www.cnblogs.com/CocoML)
> 

### 0. Setup Environment

```jsx
(opensora) zhaotianchen@is-c7bbldidqx6mb3fj-devmachine-0:~$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Aug_15_22:02:13_PDT_2023
Cuda compilation tools, release 12.2, V12.2.140
Build cuda_12.2.r12.2/compiler.33191640_0

---
(opensora) zhaotianchen@is-c7bbldidqx6mb3fj-devmachine-0:~$ g++ --version
g++ (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0
Copyright (C) 2019 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```

- 测试命令放在infini-ztc-1； `~/project/cuda_learning/`

```jsx
nvcc hello_world.cu -o hello_world
```

### 1. Simple VecSum Example

- CPU Example: `vec_sum_cpu.cpp`
    - 分配内存  `static_cast<float*>(malloc(mem_size))`
        - 意味着首先使用`malloc`分配内存，然后将返回的`void*`指针转换为`float*`类型，以便可以将其作为浮点数数组来使用
        - **static_cast<float*>**: 这是C++中的类型转换操作符，用于将一个指针转换成另一种类型的指针。`static_cast`不会进行任何类型的检查或转换，它只是告诉编译器将一个指针的类型转换为另一个指针的类型。在这个例子中，它将`malloc`返回的`void*`指针转换为`float*`类型，即指向`float`的指针。

```jsx
g++ vec_sum_cpu.cpp -o vec_sum_cpu
```

- GPU Example: `vec_sum_gpu`
    - GPU上的函数添加 `__**global__**`修饰符号，标识该function是在Device上的
    - `<<<M,T>>>` 在Host端启动CUDA程序的形式，M表示一个grid的block数目，T表示一个blokc并行的thread数目：
        - `add_kernel<<<1, 1>>>(cuda_x, cuda_y, cuda_out, N);`
    - `cudaMalloc((*void***)&cuda_x, mem_size);`
        - 对应文档中的说明：`__host____device__[cudaError_t](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1gf599e5b8b829ce7db0f5216928f6ecb6) [cudaMalloc](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1g37d37965bfb4803b6d4e59ff26856356) (void** devPtr, size_t size )`   Allocate memory on the device.
    - **内存释放**: 使用 `cudaMalloc` 分配的内存需要使用 `cudaFree` 函数来释放，以避免**内存泄漏**。

### 2. NVProf(Nsys) 简单Profile Kernel性能

- 似乎在最新的版本nvprof已经不支持了，对于compute capability较高的新机器…

```jsx
nvprof ./vec_sum_gpu
```

```jsx

======== Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
                  Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
                  Refer https://developer.nvidia.com/tools-overview for more details.
```

- 改用nsys，也支持命令行执行，能涵盖之前nvprof的功能：
    - 在 `~/project/profiling`中有下载好的Nsight的dpkg，安装后可用nsys调用
    - 更多的命令行功能  `nsys profile —help`
        - [User Guide — nsight-systems 2024.4 documentation (nvidia.com)](https://docs.nvidia.com/nsight-systems/UserGuide/index.html)
    - 目前的profile方式  `nsys profile —stats=true ./vec_sum_gpu`
        - 似乎nsys提供了一种兼容老版本nvprof command的子命令 `nsys nvprof ./vec_sum_gpu` 可以给出类似的结果。
        - cuda api的调用开销：绝大多数时间在等候cudaMemCpy & cudaMealloc
    
    ```jsx
    [5/8] Executing 'cuda_api_sum' stats report
    
     Time (%)  Total Time (ns)  Num Calls    Avg (ns)      Med (ns)    Min (ns)    Max (ns)     StdDev (ns)            Name
     --------  ---------------  ---------  -------------  -----------  ---------  -----------  -------------  ----------------------
         82.3      732,084,456          3  244,028,152.0  3,793,482.0  3,721,923  724,569,051  416,160,627.6  cudaMemcpy
         17.6      156,475,117          3   52,158,372.3    176,494.0    149,619  156,149,004   90,058,529.8  cudaMalloc
          0.1          452,890          3      150,963.3    108,418.0    102,467      242,005       78,900.5  cudaFree
          0.0          204,640          1      204,640.0    204,640.0    204,640      204,640            0.0  cudaLaunchKernel
          0.0           17,775          1       17,775.0     17,775.0     17,775       17,775            0.0  cudaDeviceSynchronize
          0.0            2,668          1        2,668.0      2,668.0      2,668        2,668            0.0  cuModuleGetLoadingMode
    ```
    
    - cuda Kernel的开销
    
    ```jsx
    [6/8] Executing 'cuda_gpu_kern_sum' stats report
     Time (%)  Total Time (ns)  Instances    Avg (ns)       Med (ns)      Min (ns)     Max (ns)    StdDev (ns)                     Name
     --------  ---------------  ---------  -------------  -------------  -----------  -----------  -----------  ------------------------------------------
        100.0      704,725,249          1  704,725,249.0  704,725,249.0  704,725,249  704,725,249          0.0  add_kernel(float *, float *, float *, int)
    
    [7/8] Executing 'cuda_gpu_mem_time_sum' stats report
     Time (%)  Total Time (ns)  Count    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)           Operation
     --------  ---------------  -----  ------------  ------------  ----------  ----------  -----------  ----------------------------
         72.4       19,244,480      1  19,244,480.0  19,244,480.0  19,244,480  19,244,480          0.0  [CUDA memcpy Device-to-Host]
         27.6        7,342,076      2   3,671,038.0   3,671,038.0   3,644,622   3,697,454     37,357.9  [CUDA memcpy Host-to-Device]
    ```
    
    - Memory(time/size)
    
    ```jsx
    [8/8] Executing 'cuda_gpu_mem_size_sum' stats report
    
     Total (MB)  Count  Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation
     ----------  -----  --------  --------  --------  --------  -----------  ----------------------------
         80.000      2    40.000    40.000    40.000    40.000        0.000  [CUDA memcpy Host-to-Device]
         40.000      1    40.000    40.000    40.000    40.000        0.000  [CUDA memcpy Device-to-Host]
    
    ```
    

### 3. 多线程样例：并行计算向量和

- 将上面 `<<<1,1>>>`的kernel改为多线程版本，并行多个thread进行计算
    - 改写在了 `vec_sum_gpu_T256.cu` 中
- CUDA提供了**内建变量**来访问每个thread的相关信息（在某个Grid中索引线程）：
    - `threadIdx.x`: 指此线程在`thread block`中的下标位置
    - `blockDim.x`: 指一个`thread block`中的线程数
- 若要将某个计算workload，切分到256个线程中进行计算，可设置stride：
    - 读”当前的thread“在block中的下标位置，以block中的thread个数为stride（在外面add_kernel的之后改成 `<<<1,256>>>`，调用一个block，block中有256个thread）
    
    ```jsx
    __global__ void add_kernel(float *x, float *y, float *out, int n){
        int index = threadIdx.x;  // 当前线程所在的下标位置
        int stride = blockDim.x;  // 此样例中为 256，由<<<1, 256>>>自动传递过来
    
        for (int i = index; i < n; i += stride) {
            out[i] = x[i] + y[i];
        }
    }
    ```
    
    ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled.png)
    

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%201.png)

- 延迟(256 threads): 22ms

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%202.png)

- 延迟（原始版本）：711 ms

---

- 不仅用更多的thread，也同时调用更多的block进行计算
    - 修改add_kernel让每个thread只是进行一个小小的加法
    
    ```jsx
      3 __global__ void add_kernel(float *x, float *y, float *out, int n){
      4     int tid = blockIdx.x * blockDim.x + threadIdx.x;
      5
      6      if (tid < n) {  // due to N is not divisible by block_size, some blocks need to run empty
      7         out[tid] = x[tid] + y[tid];
      8     }
      9 }
    ```
    
    - 修改Kernel Launching的配置
    
    ```jsx
     37     int block_size = 256;
     38     int grid_size = (N+block_size)/block_size;
     39     add_kernel<<<grid_size, block_size>>>(cuda_x, cuda_y, cuda_out, N);
    ```
    

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%203.png)

### 4. 多线程原理：

- GPU 上一般包含很多流式处理器 SM（Stream Processor，比如V100就有80个SM，SM中包含了CUDA Core），SM可以看做是**基本计算单元**，其中切分成了若干个Grid，每个Grid中包含了若干个线程块（Block，比如65536），每个线程块包含了若干线程（Thread，如512）。
    - Thread计算的基本单位，一个CUDA Kernel（workload）可以被多个Thread（实际的硬件单元算力抽象）来执行
    - Block：由多个Thread组成，同一个block中的threads可以**互相同步**，并且可以访问Shared Memory
    - Grid：多个Blocks可以组成一个Grid
    - **注意：**Block和Threads的排布方式可以是1-D，2-D，3-D的
- 线程唯一标识编号，计算方式和block/thread的dimension数目有关
    - 比如Grid一维，Block为二维：
        
        ```jsx
        int threadId = blockIdx.x * blockDim.x * blockDim.y + 
                      threadIdx.y * blockDim.x + threadIdx.x;  
        ```
        
- Warp：用来执行同一个指令的一组thread（SIMD），典型大小为32
- 给了一个打印线程ID的example来了解机制：  `./print_id.cu`
    - 准备了128个thread
    - 设置warp_size为32，num_threads为64，num_blocks为2
    - 预先malloc每个array有128个int变量
    - 在where_is_my_id中读取内建变量的值（threadIdx, blockIdx），赋值给warp与calc_thread等数组，丢回来
    - **打印出来的结果可知：**
        - block → warps → threads
- CUDA的层次结构：
    - 3层：Grid → Block → Thread (Warp是一个软件概念，可以在线程调度的过程中动态组成成新的thread)，一个3维的Example：
        
        ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%204.png)
        
    - 在每次调用CUDA Kernel的时候，都会实例化一个新的Grid，其中Block等size的大小可以用blockDim的变量进行配置。
    
    ```jsx
    #define CEIL_DIV(M, N) (((M) + (N)-1) / (N))  // M+N/N表示Ceil，分子上额外-1
    
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));  // CEIL_DIV 向上取整的除法
    // 32 * 32 = 1024 thread per block
    dim3 blockDim(32, 32);
    ```
    

### 5.1  简单矩阵乘法Example：

- 以下是进行一个朴素的GEMM实现的配置方式  `simple_matmul.cu`
    - 矩阵 $A \in [M,K]$
    - 矩阵 $B \in [K,N]$
    - CPU的三层循环计算，作为Reference： `sgemm_naive_cpu`
        - M,N,K循环三轮; A[m,k]*B[k,n]
            - 在K维度上reduce sum起来，赋值给输出矩阵的[m,n]
    - 一个朴素的gemm kernel  `sgemm_naive_kernel`
        - 如果(m<M, n<N)，在M，N维度上parallel，一层循环K
            - 二维：block*thread = ((M,32),32) = M

### 5.2  优化矩阵乘法Example-1（使用SharedMemory）：

> 更多有关Shared Memory的资源：[Using Shared Memory in CUDA C/C++ | NVIDIA Technical Blog](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)
> 
- **To-Learn：**如何使用共享内存块（SMEM），去将全局内存中局部的一部分加载到共享内存中进行计算，以**减少对全局内存的访问次数，用Shared Memory的访问来替代**。
    - 每个SM都有一块单独的共享内存；
    - 共享内存块的大小是可以配置的，与L1缓存共用。
- **思路：将局部的32x32的数据给Cache到Shared Memory中，在一个Block内部进行并行计算**
    - Load Data: 将32x32的数据从Global Memory读取到Shared memory当中
    - 在当前缓存块上进行点积（乘法并累加）
    - 推动指针取下一个cache block

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%205.png)

- Code Read:
    - 对Kernel声明一个全局变量
        - 在 CUDA 的 `__global__` 函数中使用 `template` 而是为了在定义在编译时参数化内核函数的常量值
    
    ```jsx
    template <const int BLOCKSIZE>
    ```
    
    - 声明Shared Memory：
        - `__shared__` 关键字
    
    ```jsx
        // allocate shared memory for the input and output submatrices
        __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
        __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];
    ```
    
    - 计算”每个block内部”，当前thread正在访问的数据位置
    
    ```jsx
        // the inner row & col that we're accessing in this thread
        const uint thread_row = threadIdx.x / BLOCKSIZE;
        const uint thread_col = threadIdx.x % BLOCKSIZE;
    ```
    
    - 依据当前block的idx，在输入输出数据（矩阵ABC）中寻址对应的位置：
    
    ```jsx
        // the output block that we want to compute in this threadblock
        const uint c_row = blockIdx.x;
        const uint c_col = blockIdx.y;
        
        // advance pointers to the starting positions
        A += c_row * BLOCKSIZE * K; // A \in [M,K]
        B += c_col * BLOCKSIZE;     // B \in [K,N]
        C += c_row * BLOCKSIZE * N + c_col * BLOCKSIZE;  // C \in [M,N]
    ```
    
    - 在K维度上进行for loop, 其中间隔为Block_SIZE，把原本在K维度上迭代，拆分成两层：
        - 在`K//BLOCK_SIZE`，以及在`BLOCK_SIZE`内部的迭代 （注意在这个例子中，并没有处理越界的情况，而是假设了`K`能够整除`BLOCK_SIZE`, `256//32` ）
        
        ```jsx
        for (int i = 0; i < K; i += BLOCKSIZE)
        ```
        
        - 读取对应块的数据
        
        ```jsx
        // load the next block of the input matrices into shared memory
        A_shared[thread_row * BLOCKSIZE + thread_col] = A[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B[thread_row * N + thread_col];
        ```
        
        - 同步  `__syncthreads();`
        - 进行局部的累加，循环`BLOCK_SIZE`次：
        
        ```jsx
        for (int j = 0; j < BLOCKSIZE; j++)
        {
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[j * BLOCKSIZE + thread_col];
        }
        ```
        
        - 同步  `__syncthreads();`
        - 调整指针的位置（其实也可以用循环变量i）
        
        ```jsx
        // advance the pointers
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;
        ```
        
        ---
        
        - (*) 当一个Block内部的thread所执行的内容并不是完全可以并行时，需要在Block内部进行**”同步“**，使用 `__syncthreads()`，在该例子中：
            - （1）读取数据到shared memory之后需要同步
            - （2）利用shared memory进行局部partial sum之后也需要同步。
- **全流程的示意图：**

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled.jpeg)

- 例：A100的内存层次结构图示

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%206.png)

### 5.2  优化矩阵乘法Example-2（使用1-D Thread TIle）：

- 上面例子中用一个thread来对应[M,N]中的输出数据，并行度太高，导致”算术强度“（计算量/内存搬运量）过低。
- 让一个Thread计算更多数据可以增加Arithmetic Intensity的原因：
    - 至少这个例子中，读到的一些数据可以在计算中复用*（读2*7个数据，可以计算得到4个而不是2个值）

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%207.png)

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%208.png)

- **1维Thread Tile并行：**将Thread Block中的Thread一维划分，每个Thread负责一部分而不是一个数据
    - 可以减少Block的个数，减少Block的同步开销
- **Code Read:**
    - 新增一个内循环，每个Thread计算多个条目，缓存不再是一个`BLOCK_SIZE x BLOCK_SIZE`的，而是 `BMxBK+BNxBK`
    - 本来在最内层的只有`BK`维度的循环，每个thread出一个数字，现在改为了每个threadRow(对应A的一条数据)算出`1xTM` 个数字（图中阴影的橙色和粉色的）
    - 用  `for (uint res_idx = 0; res_idx < TM; res_idx++)`  的一重循环
    - 某个Thread局部的值存储要存在register_file里，不然默认写到Shared/Global Memory中
    
    ```jsx
     float thread_results[TM] = {0.0};
    ```
    
- 在kernel定义的时候就指定了模板参数，调用的时候也输入进去：

```jsx
// ---------- Definition --------
template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(float *A, float *B, float *C, int M, int N, int K)

// ------------ Actual Calling ------------
    sgemm_blocktiling_1d_kernel<BM, BN, BK, TM>
        <<<grid_size, block_size>>>(A, B, C, m, n, k);
```

- 全流程图示：

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%209.png)

- 一些其他关于GEMM优化的文档：
    - [CUDA 矩阵乘法终极优化指南 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/410278370)
    - https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE

### 6.1 Reduce实现Example

- `”Reduce”` （规约）作为一个高阶函数(high-order function)：
    - [Fold (higher-order function) - Wikipedia](https://en.wikipedia.org/wiki/Fold_(higher-order_function))
    - 将某个数据结构recursively，采用一个Combine函数将其聚合为一个值（以下例子中我们用sum作为这个聚合，数据结构为List，退化为一个最简单的array求和的问题）
- CPU的简单实现：

```jsx
// reduce cpu version
int reduce(int *arr, int len) {
    int sum = 0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum;
}
```

- GPU实现：
    - 第一轮计算中：奇数线程将自己的值累加到偶数线程中
    - 第二轮：（每4个一组中）第0个和2个累加到第0个中
    - 依次以`2^N` 循环，直到 `BLOCK_SIZE`  (`bdim`)
    - 所有Block累加

```jsx
    // 每个线程计算 log2(bdim)-1 个轮回
    // 比如 bdim = 8, 则每个线程计算 2 个轮回
    for (int s = 1; s < bdim; s *= 2)
    {
        if (tid % (2 * s) == 0 && i + s < len)
        {
            sdata[tid] += sdata[tid + s];
        }
        // 等待所有线程完成 后再进行下一轮计算
        __syncthreads();
    }
```

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2010.png)

### 6.2 Reduce改进：交错寻址（Interleaved Addressing）

- 上述朴素Code的关键问题：
    - Warp Divergent 同一个Warp中在执行不同指令，会导致部分指令被阻塞（一个Warp，包含32个Thread，一个Warp中的线程执行相同的指令，要么执行if要么执行else分支，详细原理见 `“7. CUDA执行模型”`）
    - 取模操作 `%` 本身开销大：
        - 包含除法，除法**较难并行化**，其他一些线程等待除法线程，导致并行度下降
        - 取模操作伴随着内存访问，会导致不规则的内存访问模式
- 之前的代码里面我们是基于线程的 id 来进行寻址的，偶数线程的行为和奇数线程的行为是不一样的，导致了**一半的线程被阻塞，没有用上**。

```jsx
// 不使用交错寻址
for (int s = 1; s < bdim; s *= 2)
{
    if (tid % (2 * s) == 0 && i + s < len)
    {
        sdata[tid] += sdata[tid + s];
    }
    // 等待所有线程完成 后再进行下一轮计算
    __syncthreads();
}

// 使用交错寻址
for (int s = 1; s < bdim; s *= 2)
{
    int index = 2 * s * tid;
    if ((index + s < bdim) && (bdim * bid + s < len))
    {
        sdata[index] += sdata[index + s];
    }
}
```

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2011.png)

（始终只有偶数的Thread在进行计算，而奇数线程阻塞）

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2012.png)

当BLOCK_SIZE为256时，一个Block包含了8个Warp（32个线程组，Warp_idx如图中序号所示）（前三次迭代的时候不存在Warp Divergent，第三次迭代依然能用满一个Warp），**”让连续的线程尽量保持一样的行为“**

- （？） Warp Divergent导致慢的本质原因？在上述的例子中， 仍然有一半的线程会是IDLE的？

### 6.2 Reduce改进：解决Bank Conflict

- Bank：Shared Memory的最小单元，如果多个thread访问同一个bank，他们的访问是串行化的，从而导致阻塞。
    - 共享内存逻辑上被分为32个bank
    - 避免冲突的典型方法：改变数据布局，使用padding，shuffle指令等。
- 上述交叉寻址的方式带来了新的问题”bank conflict“：当同一个warp中多个线程访问同一个bank的时候，出现冲突。
    - **具体的：**0号线程加载shared memory中的0,1,写回0；第16号线程加载32,33，写回32。在这一个warp内访问了一个bank的不同地址（1和33号位置）
        
        ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2013.png)
        
    - **解决方案：**让一个Warp内的线程不是同一个Bank
        
        ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2014.png)
        
- 如何检测到Bank Conflict:
    - bank conflict可以在nvprof工具中进行查看（对nsys似乎不适用）：
        
        ```jsx
        nsys nvprof --events shared_st_bank_conflict ./reduce_interleaved_addressing
        
        // The --events shared_st_bank_conflict switch is ignored by nsys.
        
        ```
        
    - 只能使用Nsight Compute了
        
        ```jsx
        sudo ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum  ./reduce_interleaved_addressing
        ```
        

## 7. CUDA执行模型

- SM（Stream Processor）架构图：
    - CUDA Core 计算核心
    - 共享内存/一级缓存（Shared Memory/L1 Cache）
    - 寄存器文件（Register File）
    - 加载存储单元（Load/Store Unit）
    - 特殊功能单元（SFU, Special Function Unit）
    - 现成束调度器（Warp Scheduler）

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2015.png)

- 软硬件对应模型

![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2016.png)

- 软硬件对应分配关系：
    - 当一个Thread Block被分配到某个SM中执行之后，他就只能在这个SM中执行了
    - 线程束（Warp）：当将block分配给SM之后，SM将该block划分为多个warp，以供硬件调度。
    - 线程块Block实质上是逻辑产物，在硬件中是不存在的；而内存是一维线性存在的，block的引入是为让程序更容易被理解，因为经常处理的是图像，因此用3维的Block会更容易理解。
- CUDA对Thread的管理方式：**SIMT**（单指令，多线程），在一个线程束（Warp，32个thread为一组）中的所有thread同时执行相同的指定，但是每个线程处理的数据不同，有自己的指令地址计数器和寄存器。
- Magic Number: 32, 由硬件决定的。实质是SM用SIMD方式处理线程时的**工作粒度**。
- **线程同步机制：**当多个线程以未定义顺序访问同一数据，可能会造成不可预测的行为，CUDA内部有同步机制，但是Block之间的同步，需要显示的利用同步原语 `__synthreads()`
- **资源分配问题：**
    - 每个SM上有32位的寄存器，与一定量的共享内存来分配。每个现成需要的寄存器越多，那么SM上可运行（活跃的）Thread就越少。
- **最大化活跃Warp数：**为了更高的利用率
    - 当计算资源被分配给某个Block时，该block中所包含的warp就是活跃的warp。（？）这样warp scheduler才有空间进行schedule。
- 隐藏延迟：尽量保持同一时间，即使有部分线程被阻塞，还有别的线程可以被执行
    
    ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2017.png)
    
    - 指令的延迟分为两种：**算术延迟**（一个算术操作从开始，到产生结果之间的时间） & **内存延迟**（产生内存访问的时候，计算单元要等数据从内存拿到寄存器）
    - 只要线程束足够多，那么就可以隐藏更多的延迟。**那么至少需要多少线程，线程束来保证最小化延迟呢？  利特尔法则（Little’s Law）**
    
    ```jsx
    所需线程束=延迟×吞吐量
    ```
    
    - 假设Kernel中某条指令的延迟是5个周期，为了保持在周期内执行6个warp的吞吐率，需要30个未完成的Warp。
- 线程束调度（Warp Scheduler）和指令调度单元（Instruction Dispatch Unit）
    - 每个SM有若干个（以2个为例）
    - 2个Warp Scheduler选择2个Warp，用指令调度器存储两个Warp所对应的指令
    
    ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2018.png)
    
    - 每个warp在同一时刻执行一个指令，而同一个block之间切换warp是没有时间消耗的overhead的，因此，GPU支持”并发执行Kernel“，可以并发一些小的kernel来充分利用GPU。
    
    ![Untitled](%5BCUDA%20Learning%5D%20fe71681b7adc4709871b2af69c765163/Untitled%2019.png)
    

---

# Digest of: Nvidia’s CUDA Basics

> 笔记：[CUDA.pdf (iitk.ac.in)](https://www.cse.iitk.ac.in/users/biswap/CASS18/CUDA.pdf)
> 
- Compute Compatibility: 用来描述特定平台架构所支持的Feature
- 当前Kernel能掌握多少资源优内建变量：
    - blockDim, GridDim所定义
- Textures: 是一个ReadOnly的，物理上的Cache结构

# [Application] 将CUDA Code打包为Python Extension的形式

> 主要参考了PyTorch官方的Tutorial：https://pytorch.org/tutorials/advanced/cpp_extension.html
> 

> 官方的样例为：[pytorch/extension-cpp: C++ extensions in PyTorch (github.com)](https://github.com/pytorch/extension-cpp/tree/master)
> 

> （本地化）样例为[A-suozhang/diffuser-dev at guyue/mixdq_demo (github.com)](https://github.com/A-suozhang/diffuser-dev/tree/guyue/mixdq_demo)
> 

> 此前的笔记：[‍⁠⁠‌‬‬‬‌‬‌‬‬⁠﻿‍​‬‬[MixDQ 加速Demo] - 飞书云文档 (feishu.cn)](https://infinigence.feishu.cn/wiki/BtZLwOyYniUg80k1jPTcUANrn1f)
> 

### Installation

- 在本地的`setup.py`中定义了如何编译出wheel并安装，通过 `pip install .`  进行安装
    - `get_extension()` 中描述了从什么路径fetch所有的cc以及cu文件，编译args，以及包的名字与版本说明
        - 包含了extension_dir中的cc,cpp,cu文件
        - 包含了cutlass的header和tools文件
        - 最后pack成一个`CUDAExtention`类(`torch.utils.cpp_extension`) 中包含的文件输入给setup函数
    - 完成安装后会出现一个 `${package_name}.egg-info` 是python package安装所产生的，里面包含了一些记录dependency等metadata的文本文件
    - `setup()` 为main函数，其描述的过程：[Learn about Building a Python Package — Python Packaging Guide (pyopensci.org)](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-distribution-files-sdist-wheel.html)
        - 
    - 编译完成的文件在 ./build/ 中（若干MB）
        - `temp.linux-x86_64-cpython-38` 中包含了各种.o的输出文件
            - 以及Ninja的一些build_log
        - `lib.linux-x86_64-cpython-38` 中包含了动态链接库.so
- **依赖项CUTLASS:** 写在了项目的 `.gitmodule` 中，可以通过`git clone —recursive`直接下载下来，或者手动进行 `git submodule int & git submodule update`

### 打包成PyTorch所支持的Customize Extension

- **在setup.py中配置**，以在安装时候编译产生whl：

```jsx
from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='lltm_cpp',
      ext_modules=[cpp_extension.CppExtension('lltm_cpp', ['lltm.cpp'])],
      cmdclass={'build_ext': cpp_extension.BuildExtension})
```

- **撰写C++ Op：** (对于我们的cuda kernel来说，主要加一个C++ wrapper即可)
    - 例子： `csrc/qliner/qlinear.cc`

```jsx
#include <torch/extension.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

-----

at::Tensor 
qlinear_w8_a8_ohalf(const at::Tensor input_int8,
                    const at::Tensor weight_int8,
                    const at::Tensor weight_scale,
                    const at::Tensor input_scale,
                    const at::Tensor input_zero_point,
                    const at::Tensor weight_sum_by_input_channels,
                    const at::Tensor scale,
                    const at::Tensor bias0,
                    at::optional<const at::Tensor> bias)
{}
```

- `<torch/extension.h>` is the one-stop header to include all the necessary PyTorch bits to write C++ extensions
    - 包含了pybind和ATen（Pytorch的主要tensor computation）所需要的headers
- **Binding to Python**

```
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lltm_forward, "LLTM forward");
  m.def("backward", &lltm_backward, "LLTM backward");
}
```

- (?) 有一个section是**using accessor**，似乎是一个更高维度的类Tensor封装，而不是直接

### mixdq_extension代码

> 除了csrc/中，都是python code，在`ops/`和`nn/`文件夹中添加了空白的`__init__.py`以被识别为package引用，主体内容其实就是test功能性。
> 
- ops：
    - [`quant.py`](http://quant.py) ：对比了`torch.quantizer_per_tensor` 与 `mixdq_extension._C.quantize_per_tensor_to_int8`  的结果（在默认，以及cuda_graph的setting下）
    - `qlinear.py`：测试了torch的实现与`qlinear = mixdq_extension._C.qlinear_w8_a8_ohalf`
    - [`qconv2d.py`](http://qconv2d.py)：测试了 `mixdq_extension._C.qconv2d_w8_a8_ohalf` 和 F.conv2d
- nn: (兼容pytorch quantization的过程，符合仿真代码)
    - `utils.py`: 定义了QParams，int4的各种处理
        - conv2d_on_quantized_data （Depreacted）
            - conv_cutlass
        - linear_on_quantized_data（Deprecated）
            - gemm_cutlass
    - `Conv2d.py/Linear.py`: QuantizedConv2d
        - `from_float()`
        - `forward()` 调用了qconv2d
        - `forward_callback()` 仍然用F.conv2d
    - `quantizer_dequantizer.py`  （？是否有被用到）
- csrc：
    - 

### 外围Python Code

### Qs

- [ ]  开发过程中，大概不会用python调用来对比reference的吗？
    - ops里面的code是调用的`mixdq_extension._C.qconv2d_w8_a8_ohalf` 来操作，如果这样的话每次修改code都需要跑一个完整的编译过程（是不是有点费劲了）
    - debug的时候直接写个main函数nvcc单个文件的吗
- [ ]  profiling工具？
- [x]  conv2d_on_quantized_data 是在哪里被用到？还是只是test用？

# References:

- [🌟]中文的材料，PaddleJiTLab：[CUDATutorial | Notebook (keter.top)](https://cuda.keter.top/)
    - https://github.com/PaddleJitLab/CUDATutorial
    - 深度学习基础知识目录 | Notebook (keter.top) 格式参考了这个博客，前端很美观
        - [CUDA执行模型概述 | Notebook (keter.top)](https://space.keter.top/docs/high_performance/CUDA%E7%BC%96%E7%A8%8B/CUDA%E6%89%A7%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0)
- https://infinigence.feishu.cn/wiki/BtZLwOyYniUg80k1jPTcUANrn1f MixDQ Demo的构建过程
- [Tutorial 01: Say Hello to CUDA - CUDA Tutorial (cuda-tutorial.readthedocs.io)](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/#:~:text=Tutorial%2001%3A%20Say%20Hello%20to%20CUDA%201%20Introduction,...%207%20Wrap%20up%20...%208%20Acknowledgments%20)
    - https://developer.nvidia.com/blog/even-easier-introduction-cuda/ 的改进版本，用ReadTheDoc改写了两个煎蛋的Example
- https://github.com/RussWong/CUDATutorial 一个国人给了一系列NN相关的例子，最后包含了 fused kernel
- [CUDA C++ Programming Guide (nvidia.com)](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- A simple short Slides: [CUDA.pdf (iitk.ac.in)](https://www.cse.iitk.ac.in/users/biswap/CASS18/CUDA.pdf)
- PDF on Github: 《CUDA Programming: A Developer’s Guide》 书籍
    - [cudaLearningMaterials_2/CUDA并行程序设计-GPU编程指南-高清扫描-中英文/CUDA Programming A Developer's Guide to Parallel Computing with GPUs.pdf at master · SeventhBlue/cudaLearningMaterials_2 (github.com)](https://github.com/SeventhBlue/cudaLearningMaterials_2/blob/master/CUDA%E5%B9%B6%E8%A1%8C%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1-GPU%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E9%AB%98%E6%B8%85%E6%89%AB%E6%8F%8F-%E4%B8%AD%E8%8B%B1%E6%96%87/CUDA%20Programming%20A%20Developer's%20Guide%20to%20Parallel%20Computing%20with%20GPUs.pdf)
- 知乎Example: [CUDA 编程小练习（目录） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365904031)
- Nvidia Developer Blogs的对应Code：[code-samples/posts at master · NVIDIA-developer-blog/code-samples (github.com)](https://github.com/NVIDIA-developer-blog/code-samples/tree/master/posts)
    - 一些新Feature的介绍：[Programming Tensor Cores in CUDA 9 | NVIDIA Technical Blog](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
- 一系列库的简介：[cuBLAS - 上海交大超算平台用户手册 Documentation (sjtu.edu.cn)](https://docs.hpc.sjtu.edu.cn/app/compilers_and_languages/cublas.html)