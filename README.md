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
    - GPU上的函数添加 __**global__**修饰符号
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
    - Block：由多个Thread组成，同一个block中的threads可以互相同步，并且可以访问Shared Memory
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
    

- 以下是进行一个朴素的GEMM实现的配置方式  `simple_matmul.cu`
    - 矩阵 $A \in [M,K]$
    - 矩阵 $B \in [K,N]$
    - CPU的三层循环计算，作为Reference： `sgemm_naive_cpu`
        - M,N,K循环三轮; A[m,k]*B[k,n]
            - 在K维度上reduce sum起来，赋值给输出矩阵的[m,n]
    - 一个朴素的gemm kernel  `sgemm_naive_kernel`
        - 如果(m<M, n<N)，在M，N维度上parallel，一层循环K
            - 二维：block*thread = ((M,32),32) = M

### References:

- [🌟]中文的材料，PaddleJiTLab：[CUDATutorial | Notebook (keter.top)](https://cuda.keter.top/)
    - https://github.com/PaddleJitLab/CUDATutorial
    - 深度学习基础知识目录 | Notebook (keter.top) 格式参考了这个博客，前端很美观
        - [CUDA执行模型概述 | Notebook (keter.top)](https://space.keter.top/docs/high_performance/CUDA%E7%BC%96%E7%A8%8B/CUDA%E6%89%A7%E8%A1%8C%E6%A8%A1%E5%9E%8B%E6%A6%82%E8%BF%B0)
- https://infinigence.feishu.cn/wiki/BtZLwOyYniUg80k1jPTcUANrn1f MixDQ Demo的构建过程
- [Tutorial 01: Say Hello to CUDA - CUDA Tutorial (cuda-tutorial.readthedocs.io)](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/#:~:text=Tutorial%2001%3A%20Say%20Hello%20to%20CUDA%201%20Introduction,...%207%20Wrap%20up%20...%208%20Acknowledgments%20)
    - https://developer.nvidia.com/blog/even-easier-introduction-cuda/ 的改进版本，用ReadTheDoc改写了两个煎蛋的Example
- https://github.com/RussWong/CUDATutorial 一个国人给了一系列NN相关的例子，最后包含了 fused kernel
- A simple short Slides: [CUDA.pdf (iitk.ac.in)](https://www.cse.iitk.ac.in/users/biswap/CASS18/CUDA.pdf)
- PDF on Github: 《CUDA Programming: A Developer’s Guide》 书籍
    - [cudaLearningMaterials_2/CUDA并行程序设计-GPU编程指南-高清扫描-中英文/CUDA Programming A Developer's Guide to Parallel Computing with GPUs.pdf at master · SeventhBlue/cudaLearningMaterials_2 (github.com)](https://github.com/SeventhBlue/cudaLearningMaterials_2/blob/master/CUDA%E5%B9%B6%E8%A1%8C%E7%A8%8B%E5%BA%8F%E8%AE%BE%E8%AE%A1-GPU%E7%BC%96%E7%A8%8B%E6%8C%87%E5%8D%97-%E9%AB%98%E6%B8%85%E6%89%AB%E6%8F%8F-%E4%B8%AD%E8%8B%B1%E6%96%87/CUDA%20Programming%20A%20Developer's%20Guide%20to%20Parallel%20Computing%20with%20GPUs.pdf)
- 知乎Example: [CUDA 编程小练习（目录） - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/365904031)