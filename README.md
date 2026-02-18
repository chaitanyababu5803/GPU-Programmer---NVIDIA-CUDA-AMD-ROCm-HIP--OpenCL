CUDA Programming
*****************************
https://www.geeksforgeeks.org/electronics-engineering/introduction-to-cuda-programming/
***********************************
In this article, we will cover the overview of CUDA programming and mainly focus on the concept of CUDA requirement and we will also discuss the execution model of CUDA. Finally, we will see the application. Let us discuss it one by one.

CUDA stands for Compute Unified Device Architecture. It is an extension of C/C++ programming. CUDA is a programming language that uses the Graphical Processing Unit (GPU). It is a parallel computing platform and an API (Application Programming Interface) model, Compute Unified Device Architecture was developed by Nvidia. This allows computations to be performed in parallel while providing well-formed speed. Using CUDA, one can harness the power of the Nvidia GPU to perform common computing tasks, such as processing matrices and other linear algebra operations, rather than simply performing graphical calculations.

Why do we need CUDA?
GPUs are designed to perform high-speed parallel computations to display graphics such as games.
Use available CUDA resources. More than 100 million GPUs are already deployed.
It provides 30-100x speed-up over other microprocessors for some applications.
GPUs have very small Arithmetic Logic Units (ALUs) compared to the somewhat larger CPUs. This allows for many parallel calculations, such as calculating the color for each pixel on the screen, etc.
Architecture of CUDA
<img width="1001" height="477" alt="image" src="https://github.com/user-attachments/assets/34b49b3b-2ddc-4d0e-8fea-fd7f1559f8ce" />

16 Streaming Multiprocessor (SM) diagrams are shown in the above diagram.
Each Streaming Multiprocessor has 8 Streaming Processors (SP) ie, we get a total of 128 Streaming Processors (SPs).
Now, each Streaming processor has a MAD unit (Multiplication and Addition Unit) and an additional MU (multiplication unit).
The GT200 has 30 Streaming Multiprocessors (SMs) and each Streaming Multiprocessor (SM) has 8 Streaming Processors (SPs) ie, a total of 240 Streaming Processors (SPs), and more than 1 TFLOP processing power.
Each Streaming Processor is gracefully threaded and can run thousands of threads per application.
The G80 card has 16 Streaming Multiprocessors (SMs) and each SM has 8 Streaming Processors (SPs), i.e., a total of 128 SPs and it supports 768 threads per Streaming Multiprocessor (note: not per SP).
Eventually, after each Streaming Multiprocessor has 8 SPs, each SP supports a maximal of 768/8 = 96 threads. Total threads that can run on 128 SPs - 128 * 96 = 12,228 times.
Therefore these processors are called massively parallel.
The G80 chips have a memory bandwidth of 86.4GB/s.
It also has an 8GB/s communication channel with the CPU (4GB/s for uploading to the CPU RAM, and 4GB/s for downloading from the CPU RAM).
How CUDA works?
GPUs run one kernel (a group of tasks) at a time.
Each kernel consists of blocks, which are independent groups of ALUs.
Each block contains threads, which are levels of computation.
The threads in each block typically work together to calculate a value.
Threads in the same block can share memory.
In CUDA, sending information from the CPU to the GPU is often the most typical part of the computation.
For each thread, local memory is the fastest, followed by shared memory, global, static, and texture memory the slowest.
Typical CUDA Program flow
Load data into CPU memory
Copy data from CPU to GPU memory - e.g., cudaMemcpy(..., cudaMemcpyHostToDevice)
Call GPU kernel using device variable - e.g., kernel<<<>>> (gpuVar)
Copy results from GPU to CPU memory - e.g., cudaMemcpy(.., cudaMemcpyDeviceToHost)
Use results on CPU
How work is distributed?
Each thread "knows" the x and y coordinates of the block it is in, and the coordinates where it is in the block.
These positions can be used to calculate a unique thread ID for each thread.
The computational work done will depend on the value of the thread ID.
For example, the thread ID corresponds to a group of matrix elements.

CUDA Applications
CUDA applications must run parallel operations on a lot of data, and be processing-intensive.

Computational finance
Climate, weather, and ocean modeling
Data science and analytics
Deep learning and machine learning
Defence and intelligence
Manufacturing/AEC
Media and entertainment
Medical imaging
Oil and gas
Research
Safety and security
Tools and management
Benefits of CUDA
There are several advantages that give CUDA an edge over traditional general-purpose graphics processor (GPU) computers with graphics APIs:

Integrated memory (CUDA 6.0 or later) and Integrated virtual memory (CUDA 4.0 or later).
Shared memory provides a fast area of shared memory for CUDA threads. It can be used as a caching mechanism and provides more bandwidth than texture lookup.
Scattered read codes can be read from any address in memory.
Improved performance on downloads and reads, which works well from the GPU and to the GPU.
CUDA has full support for bitwise and integer operations.
Limitations of CUDA
CUDA source code is given on the host machine or GPU, as defined by the C++ syntax rules. Longstanding versions of CUDA use C syntax rules, which means that up-to-date CUDA source code may or may not work as required.
CUDA has unilateral interoperability(the ability of computer systems or software to exchange and make use of information) with transferor languages like OpenGL. OpenGL can access CUDA registered memory, but CUDA cannot access OpenGL memory.
Afterward versions of CUDA do not provide emulators or fallback support for older versions.
CUDA supports only NVIDIA hardware.


*********************************************************
CUDA Hello World
This is a simple example of a CUDA C program that demonstrates how to execute a kernel on the GPU and print a message to the console.
Prerequisite :
NVIDIA GPU with CUDA support
CUDA Toolkit installed
Compatible C compiler (e.g., gcc)
Code :
#include <stdio.h>
#include <cuda.h>

__global__ void dkernel(){ //__global__ indicate it is not normal kernel function but for GPU
printf(“Hello world \n”);
}

int main (){
dkernel <<<1,1>>>();//<<<no. of blocks,no. of threads in in block>>>

cudaDeviceSynchronize(); //Tells GPU to do all work than synchronize GPU buffer with CPU.

return 0;

}


****************************
What is ROCm?
ROCm is a software stack, composed primarily of open-source software, that provides the tools for programming AMD Graphics Processing Units (GPUs), from low-level kernels to high-level end-user applications.
<img width="1041" height="833" alt="image" src="https://github.com/user-attachments/assets/77605adb-2b24-49e4-b363-e45f23db14fa" />

**********************
What is AMD ROCm, why was it invented and what can one do with it?
C/C++
Is it better than CUDA?
open-source software stack designed for harnessing the computational power of AMD's GPUs. It was created to provide a robust and open alternative to NVIDIA's CUDA, which has long been the dominant platform for GPU computing. By being open-source, ROCm offers several key advantages:

Flexibility and Customization

No Vendor Lock-in

Community-Driven Development

What Can You Do With It? ROCm is particularly impactful in fields that require massive parallel processing capabilities. basically everything thing that requires ai/video/image processing…

****************************
AMD ROCm (Radeon Open Compute) is a software stack. 
*********************
https://www.reddit.com/r/AskProgramming/comments/1n4avki/what_is_amd_rocm_why_was_it_invented_and_what_can/
It is an open-source platform designed for GPU computing, acting as the software layer that allows developers to program AMD GPUs for high-performance computing (HPC), artificial intelligence (AI), and machine learning (ML) tasks. 
Here is a breakdown of what AMD ROCm consists of:

1. It is a Software Stack (Not Hardware) 
ROCm provides the necessary software tools to control and utilize AMD GPU hardware. It includes: 
Drivers: The low-level interface (specifically the ROCk kernel driver) that allows the operating system to talk to the GPU.
Runtimes: The user-mode APIs (ROCr) that allow applications to launch compute kernels on the GPU.
Compilers: Tools like HIPCC (based on Clang/LLVM) that compile code for AMD GPUs.
Libraries: High-level mathematical and AI libraries (such as MIOpen, rocBLAS, and RCCL) optimized for AMD hardware.
Debuggers and Profilers: Tools to help developers troubleshoot and optimize code. 

2. Primary Function
AI and Machine Learning: It enables popular frameworks like PyTorch and TensorFlow to run on AMD GPUs.
HPC and Computing: It provides an alternative to NVIDIA's CUDA platform, supporting OpenCL, HIP (Heterogeneous-computing Interface for Portability), and OpenMP. 

3. Hardware Supported
While ROCm is software, it is designed to run on specific AMD hardware: 
Professional GPUs: AMD Instinct accelerators and Radeon Pro GPUs.
Consumer GPUs: Select AMD Radeon gaming graphics cards (e.g., RDNA 2 and newer architectures). 

Summary Table
Feature 	Description
Type	Open-source software stack (libraries, drivers, compilers)
Purpose	GPU programming, AI, ML, and High-Performance Computing
Alternative to	NVIDIA CUDA
Hardware	AMD Instinct accelerators and Radeon GPUs

****************************
Optimized GPU Software Stack
AMD ROCm™ is an open software stack including drivers, development tools, and APIs that enable GPU programming from low-level kernel to end-user applications. ROCm is optimized for Generative AI and HPC applications, and it is easy to migrate existing code into ROCm software.
<img width="1357" height="709" alt="image" src="https://github.com/user-attachments/assets/32c991d0-6569-4ed8-85b8-979ca8557572" />


***************************
<img width="960" height="503" alt="image" src="https://github.com/user-attachments/assets/5cb21574-6e4f-429d-babe-d1591d970ce4" />
<img width="1000" height="675" alt="image" src="https://github.com/user-attachments/assets/cc40e00a-5450-4164-8d70-d01d93cc79c1" />
<img width="2640" height="1036" alt="image" src="https://github.com/user-attachments/assets/9cbf19b1-d515-4d9a-bcd1-32d6682430e3" />
<img width="1108" height="870" alt="image" src="https://github.com/user-attachments/assets/ba461c49-0690-494b-8a2f-d518cc56bc17" />


********************************
<img width="275" height="183" alt="image" src="https://github.com/user-attachments/assets/4b7f14f5-ddbb-49e0-8047-b3a9715d3eee" />


************************************************
suggest core topics in rocm 2026 to get job
To secure a job in the AMD ROCm ecosystem in 2026, you should focus on the following core technical areas, as modern AI engineering now demands skills in production-ready deployment, high-performance optimization, and cross-platform portability. 

1. Heterogeneous Programming & Portability
HIP (Heterogeneous-Compute Interface for Portability): Master the HIP programming model to write code that can run on both AMD and NVIDIA GPUs.
CUDA-to-HIP Migration: Learn to use tools like hipify to port existing CUDA applications to the ROCm platform. 

2. High-Performance AI Frameworks
Framework Integration: Deepen your expertise in PyTorch for ROCm and TensorFlow, specifically focusing on kernel optimization and upstream contributions.
Model Optimization: Gain skills in quantization, pruning, and using MIOpen (AMD's deep learning library) to accelerate inference and training. 

3. Distributed Systems & Infrastructure
Scalable AI Training: Learn to manage large-scale training across clusters using tools like TorchTitan and SLURM or Kubernetes (K8s) for orchestration.
MLOps: Develop proficiency in containerization using Docker for ROCm environments and setting up CI/CD pipelines for GPU-accelerated workloads. 

4. Specialized Libraries & Kernels
Mathematical Libraries: Understand and implement operations using rocBLAS, rocFFT, and rocSOLVER for HPC and AI applications.
Custom Kernel Development: Practice writing low-level GPU kernels in C++ and assembly to maximize hardware utilization. 

5. Emerging 2026 Focus Areas
Edge & Client AI: Explore ROCm support for the AMD Ryzen AI 300/400 series and Radeon RX 9000 graphics for local LLM deployment.
Robotics Integration: Learn to integrate ROCm with ROS 2 and simulation frameworks like Gazebo for AI-driven robotics.

********************************
1. Heterogeneous Programming & Portability
HIP (Heterogeneous-Compute Interface for Portability): Master the HIP programming model to write code that can run on both AMD and NVIDIA GPUs.
CUDA-to-HIP Migration: Learn to use tools like hipify to port existing CUDA applications to the ROCm platform.


*******************
What is HIP?
The Heterogeneous-computing Interface for Portability (HIP) API, part of AMD’s ROCm platform, is a C++ runtime API and kernel language that lets developers create portable applications that run on heterogeneous systems, using CPUs and AMD GPUs from a single source code base.
<img width="560" height="560" alt="image" src="https://github.com/user-attachments/assets/055319d3-52e5-43c4-bd3a-8beb6a27f9f7" />


HIP is a thin API with little or no performance impact over coding directly in AMD ROCm.

HIP enables coding in a single-source C++ programming language, including features such as templates, C++11 lambdas, classes, namespaces, and more.

Developers can tune for performance or handle tricky cases via HIP.

ROCm offers compilers (clang, hipcc), code profilers (rocprofv3), debugging tools (rocgdb), libraries and HIP with the runtime API and kernel language, to create heterogeneous applications running on both CPUs and GPUs. ROCm provides marshalling libraries like hipFFT or hipBLAS that act as a thin programming layer over AMD ROCm and offer API compatibility with the equivalent Nvidia CUDA libraries. These libraries provide pointer-based memory interfaces and can be easily integrated into your applications.

HIP supports building and running on both AMD GPUs or NVIDIA GPUs. GPU Programmers familiar with NVIDIA CUDA or OpenCL will find the HIP API familiar and easy to use. You can quickly port your application to run on the available hardware while maintaining a single codebase. The HIPify tools, based on the clang front-end and Perl language, can convert CUDA API calls into the corresponding HIP API calls. However, HIP is not intended to be a drop-in replacement for CUDA, and developers should expect to do some manual coding and performance tuning work for AMD GPUs to port existing projects as described HIP porting guide.

HIP provides two components: those that run on the CPU, also known as host system, and those that run on GPUs, also referred to as device. The host-based code is used to create device buffers, move data between the host application and a device, launch the device code (also known as kernel), manage streams and events, and perform synchronization. The kernel language provides a way to develop massively parallel programs that run on GPUs, and provides access to GPU specific hardware capabilities.

In summary, HIP simplifies cross-platform development, maintains performance, and provides a familiar C++ experience for GPU programming that runs seamlessly.

HIP components
HIP consists of the following components. For information on the license associated with each component, see HIP licensing.

C++ runtime API
HIP provides headers and a runtime library built on top of HIP-Clang compiler in the repository Compute Language Runtime (CLR). The HIP runtime implements HIP streams, events, and memory APIs, and is an object library that is linked with the application. The source code for all headers and the library implementation is available on GitHub.

For further details, check HIP Runtime API Reference.

Kernel language
HIP provides a C++ syntax that is suitable for compiling most code that commonly appears in compute kernels (classes, namespaces, operator overloading, and templates). HIP also defines other language features that are designed to target accelerators, such as:

Short-vector headers that can serve on a host or device

Math functions that resemble those in math.h, which is included with standard C++ compilers

Built-in functions for accessing specific GPU hardware capabilities

*************************************************
Introduction to the HIP programming model
The HIP programming model enables mapping data-parallel C/C++ algorithms to massively parallel SIMD (Single Instruction, Multiple Data) architectures like GPUs. HIP supports many imperative languages, such as Python via PyHIP, but this document focuses on the original C/C++ API of HIP.

While GPUs may be capable of running applications written for CPUs if properly ported and compiled, it would not be an efficient use of GPU resources. GPUs fundamentally differ from CPUs and should be used accordingly to achieve optimum performance. A basic understanding of the underlying device architecture helps you make efficient use of HIP and general purpose graphics processing unit (GPGPU) programming in general. The following topics introduce you to the key concepts of GPU-based programming and the HIP programming model.

Hardware differences: CPU vs GPU
CPUs and GPUs have been designed for different purposes. CPUs quickly execute a single thread, decreasing the time for a single operation while increasing the number of sequential instructions that can be executed. This includes fetching data and reducing pipeline stalls where the ALU has to wait for previous instructions to finish.
<img width="522" height="317" alt="image" src="https://github.com/user-attachments/assets/165fa07c-863c-431a-abf5-3c55bfb304f8" />


************
Heterogeneous programming
The HIP programming model has two execution contexts. The main application starts on the CPU, or the host processor, and compute kernels are launched on the device such as Instinct accelerators or AMD GPUs. The host execution is defined by the C++ abstract machine, while device execution follows the SIMT model of HIP. These two execution contexts are signified by the __host__ and __global__ (or __device__) decorators in HIP program code. There are a few key differences between the two contexts:

The C++ abstract machine assumes a unified memory address space, meaning that one can always access any given address in memory (assuming the absence of data races). HIP however introduces several memory namespaces, an address from one means nothing in another. Moreover, not all address spaces are accessible from all contexts.

Looking at the gcn_cu figure, you can see that every CU has an instance of storage backing the namespace __shared__. Even if the host were to have access to these regions of memory, the performance benefits of the segmented memory subsystem are supported by the inability of asynchronous access from the host.
<img width="542" height="282" alt="image" src="https://github.com/user-attachments/assets/592fd598-b14f-471b-b601-693f2e4f54ec" />


**Single instruction multiple threads (SIMT)
The HIP kernel code, written as a series of scalar instructions for multiple threads with different thread indices, gets mapped to the SIMD units of the GPUs. Every single instruction, which is executed for every participating thread of a kernel, gets mapped to the SIMD.
<img width="553" height="292" alt="image" src="https://github.com/user-attachments/assets/f85da0f0-30d6-4b6f-8927-4744e8eba271" />

********
Hierarchical thread model
As previously discussed, all threads of a kernel are uniquely identified by a set of integral values called thread IDs. The hierarchy consists of three levels: thread, blocks, and grids.
<img width="660" height="610" alt="image" src="https://github.com/user-attachments/assets/2e565543-7de3-4e18-8989-1186495e54bc" />


****************
Memory model
<img width="611" height="321" alt="image" src="https://github.com/user-attachments/assets/9d88542b-b7e0-4dee-b12a-b8e38232c978" />


***************
Memory optimizations and best practices
<img width="542" height="297" alt="image" src="https://github.com/user-attachments/assets/ca0138eb-0cf3-440f-bd8b-88069647fc54" />


******************
<img width="566" height="299" alt="image" src="https://github.com/user-attachments/assets/84264abd-f3f3-4e22-820e-0278ac38afea" />

***************************
Understanding GPU performance
This chapter explains the theoretical foundations of GPU performance on AMD hardware. Understanding these concepts helps you analyze performance characteristics, identify bottlenecks, and make informed optimization decisions.

For practical optimization techniques and step-by-step guidance, see Performance guidelines.

Performance bottlenecks
A performance bottleneck is the limiting factor that prevents a GPU kernel from achieving higher performance. The two primary categories are:

Compute-bound: The kernel is limited by arithmetic throughput

Memory-bound: The kernel is limited by memory bandwidth


*********************
Performance bottlenecks
A performance bottleneck is the limiting factor that prevents a GPU kernel from achieving higher performance. The two primary categories are:

Compute-bound: The kernel is limited by arithmetic throughput

Memory-bound: The kernel is limited by memory bandwidth

Understanding which category applies helps identify the appropriate op

******************
Hardware implementation
This chapter describes the hardware architecture of AMD GPUs supported by HIP, focusing on the internal organization and operation of GPU hardware components. Understanding these hardware details helps you optimize GPU applications and achieve maximum performance.

Overall GPU architecture
AMD GPUs consist of interconnected blocks of digital circuits that work together to execute complex parallel computing tasks. The architecture is organized hierarchically to enable massive parallelism while managing resources efficiently.

<img width="1528" height="665" alt="image" src="https://github.com/user-attachments/assets/c4d6d1ac-df5c-4462-b1f6-2844c9d2a470" />

**********************
<img width="1523" height="306" alt="image" src="https://github.com/user-attachments/assets/807c1ae2-39cb-4a12-9c6e-878a2fc2306d" />
<img width="295" height="227" alt="image" src="https://github.com/user-attachments/assets/c1d80d62-f5a9-4bd5-b50e-1864028c8f6d" />
<img width="822" height="1073" alt="image" src="https://github.com/user-attachments/assets/e2a70d75-364f-41fd-8357-0550d62f7a56" />
<img width="826" height="372" alt="image" src="https://github.com/user-attachments/assets/0750791a-6d8e-44ce-b2c6-4f252de95767" />
<img width="846" height="156" alt="image" src="https://github.com/user-attachments/assets/f80d7602-d3cb-4969-bec1-e399385126df" />
<img width="1130" height="285" alt="image" src="https://github.com/user-attachments/assets/21be6933-1155-435b-a3c0-eb2393d58bed" />


***************************************************
Hierarchical organization
The GPU organizes compute resources in a three-level hierarchy that enables modular design and resource sharing:

Shader engines (SE): Top-level organizational units containing multiple shader arrays and shared resources

Shader arrays: Groups of compute units sharing instruction and scalar caches

Compute units (CU): Basic execution units containing the ALUs and registers for thread execution


<img width="1528" height="665" alt="image" src="https://github.com/user-attachments/assets/ca228736-c5f1-40af-8ae3-af1ba5821498" />

***************************
Using HIP runtime API
The HIP runtime API provides C and C++ functionalities to manage event, stream, and memory on GPUs. The HIP runtime uses Compute Language Runtime (CLR).

CLR contains source code for AMD ROCm’s compute language runtimes: HIP and OpenCL™. CLR includes the HIP implementation on the AMD ROCm platform: hipamd and the ROCm Compute Language Runtime (rocclr). rocclr is a virtual device interface that enables the HIP runtime to interact with different backends, such as ROCr on Linux or PAL on Microsoft Windows. CLR also includes the OpenCL runtime implementation.

The HIP runtime API backends are summarized in the following figure:
<img width="560" height="560" alt="image" src="https://github.com/user-attachments/assets/d00b9b64-bea1-47d0-826d-245e5d14034f" />

******************************
Memory management
Memory management is an important part of the HIP runtime API, when creating high-performance applications. Both allocating and copying memory can result in bottlenecks, which can significantly impact performance.

The programming model is based on a system with a host and a device, each having its own distinct memory. Kernels operate on Device memory, while host functions operate on Host memory.

The runtime offers functions for allocating, freeing, and copying device memory, along with transferring data between host and device memory.

Here are the various memory management techniques:

Coherence control

Unified memory management

Virtual memory management

Stream Ordered Memory Allocator

Memory allocation
The API calls and the resulting allocations are listed here:

Memory coherence control
API

Data location

Allocation

System allocated

Host

Pageable

hipMallocManaged()

Host

Managed

hipHostMalloc()

Host

Pinned

hipMalloc()

Device

Pinned
**********************************
Reduction
Reduction is a common algorithmic operation used in parallel programming to reduce an array of elements into a shorter array of elements or a single value. This document exploits reduction to introduce some key considerations while designing and optimizing GPU algorithms.

This document is a rejuvenation and extension of the invaluable work of Mark Harris. While the author approaches the topic with a less naive approach, reviewing some original material is valuable to see how much the underlying hardware has changed. This document provides a greater insight to demonstrate progress.

The algorithm
Reduction has many names depending on the domain; in functional programming it’s referred to as fold, in C++, it’s called std::accumulate and in C++17, as std::reduce. A reduction takes a range of inputs and “reduces” the given range with a binary operation to a singular or scalar output. Canonically, a reduction requires a “zero” element that bootstraps the algorithm and serves as one of the initial operands to the binary operation. The “zero” element is generally called identity or neutral element in the group theory, which implies that it is an operand that doesn’t change the result. Some typical use cases are: calculating a sum or normalizing a dataset and finding the maximum value in the dataset. The latter use case is discussed further in this tutorial.
<img width="481" height="681" alt="image" src="https://github.com/user-attachments/assets/18346621-c342-4c96-b2f1-d853952e11f1" />

********************
Reduction on GPUs
Implementing reductions on GPUs requires a basic understanding of the Introduction to the HIP programming model. The document explores aspects of low-level optimization best discussed through the Hierarchical thread model, and refrains from using cooperative groups.

Synchronizing parallel threads of execution across a GPU is crucial for correctness as the partial results can’t be synchronized before they manifest. Synchronizing all the threads running on a GPU at any given time is possible, however, it is a costly and intricate operation. If synchronization is not absolutely necessary, map the parallel algorithm so that multiprocessors and blocks can make independent progress and need not sync frequently.

There are ten reduction implementations in the rocm-examples, which are described in the following sections.

Naive shared reduction
The naive algorithm takes a tree-like shape, where the computational domain is purposefully distributed among blocks. In all blocks, all threads participate in loading data from persistent (from the kernel’s perspective) global memory into the shared memory. This helps to perform tree-like reduction for a single thread by writing the partial result to global, in a location unique to the block, which allows the block to make independent progress. The partial results are combined in subsequent launches of the same kernel until a scalar result is reached.
<img width="481" height="611" alt="image" src="https://github.com/user-attachments/assets/676361d1-f22e-4b0a-810f-99479e6d3872" />

**********************
HIP Graph API Tutorial
Time to complete: 60 minutes | Difficulty: Intermediate | Domain: Medical Imaging

Introduction
Imagine you are directing a movie. In traditional GPU programming with streams, you are like a director who must call “action!” for every single shot, waiting between each take. With HIP graphs, you pre-plan the entire scene sequence and then call “action!” just once to film everything in one go. This tutorial will show you how to transform your GPU applications from repeated direction to choreographed performance.

Modeling dependencies between GPU operations
Most movies in the world follow a plot where certain scenes must happen before the following scenes; otherwise the movie might not make much sense. If a scene A must happen before scenes B and C, B and C depend on A. If B and C contain different stories that (at this point) are unrelated to each other, B and C are independent and can be shown to the audience in any order. However, both scenes might be a prerequisite for the final scene D, so D depends on both of them. When you represent scenes as nodes and dependencies as edges, you can create a graph, and the graph representing your imaginary movie script will have a diamond-like shape:
<img width="441" height="521" alt="image" src="https://github.com/user-attachments/assets/6f28dfd9-10c3-4d7a-b86f-7557894fe018" />

*************************
Transitioning a CT reconstruction pipeline
In this tutorial, you will modify an existing GPU-accelerated stream-based image processing pipeline that reconstructs computer tomography (CT) data (the classic Shepp-Logan phantom [ShLo74]). The pipeline transforms raw X-ray projections into clear cross-sectional images used in medical diagnosis.
<img width="1280" height="160" alt="image" src="https://github.com/user-attachments/assets/abe53c06-6d0d-4e7f-a47b-bb42fea20fba" />

************************************
************************************************
BASIC EXMAPLE WITH CODE***************
Here is a basic "Vector Square" example demonstrating the core concepts of the HIP (Heterogeneous-Compute Interface for Portability) API. This single source code can be compiled to run on both AMD and NVIDIA GPUs. 
AMD ROCm documentation
AMD ROCm documentation
 +1
HIP Basic Example: Vector Square
This program allocates memory on both the host (CPU) and device (GPU), copies data, launches a kernel to perform an operation in parallel, and copies the results back. 
AMD ROCm documentation
AMD ROCm documentation
 +1
cpp
#include <iostream>
#include <vector>
#include "hip/hip_runtime.h"

// Macro to check HIP errors
#define HIP_CHECK(call) {                                    \
    hipError_t err = call;                                   \
    if (err != hipSuccess) {                                 \
        std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " \
                  << hipGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}

/*
 * The compute kernel: runs on the device (GPU).
 * __global__ signifies that this function is a kernel and can be called from the host.
 */
__global__ void vector_square(double* C_d, double* A_d, int N) {
    // Calculate the global thread ID
    int gid = (blockDim.x * blockIdx.x) + threadIdx.x;
    
    // Ensure the thread ID is within the bounds of the array
    if (gid < N) {
        C_d[gid] = A_d[gid] * A_d[gid];
    }
}

int main() {
    int N = 10000; // Number of elements
    size_t Nbytes = N * sizeof(double);

    // 1. Allocate host (CPU) memory
    std::vector<double> A_h(N); // Input array on host
    std::vector<double> C_h(N); // Output array on host

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        A_h[i] = static_cast<double>(i);
    }

    // 2. Allocate device (GPU) memory
    double *A_d, *C_d;
    HIP_CHECK(hipMalloc(&A_d, Nbytes));
    HIP_CHECK(hipMalloc(&C_d, Nbytes));

    // 3. Copy data from host to device
    HIP_CHECK(hipMemcpy(A_d, A_h.data(), Nbytes, hipMemcpyHostToDevice));

    // 4. Launch the kernel
    // Configure launch parameters: blocks and threads per block
    const unsigned threadsPerBlock = 256;
    const unsigned blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // hipLaunchKernelGGL is the macro used to launch kernels in HIP
    hipLaunchKernelGGL(vector_square, dim3(blocks), dim3(threadsPerBlock), 0/*dynamic shared memory*/, 0/*stream*/, C_d, A_d, N);
    
    // Check for any errors during kernel execution
    HIP_CHECK(hipGetLastError());

    // 5. Copy results back from device to host
    HIP_CHECK(hipMemcpy(C_h.data(), C_d, Nbytes, hipMemcpyDeviceToHost));

    // 6. Verify the results (optional)
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (C_h[i] != A_h[i] * A_h[i]) {
            std::cerr << "Verification failed at index " << i << std::endl;
            success = false;
            break;
        }
    }

    // 7. Free device memory
    HIP_CHECK(hipFree(A_d));
    HIP_CHECK(hipFree(C_d));

    if (success) {
        std::cout << "Program executed successfully! [__SUCCESS__]" << std::endl;
    } else {
        std::cout << "Program failed verification." << std::endl;
    }

    return 0;
}
Key HIP Concepts
__global__: A function type qualifier that indicates the function is a kernel and will execute on the GPU, callable from the CPU (host).
hipMalloc: Allocates memory on the device (GPU).
hipMemcpy: Copies data between host and device memory (e.g., hipMemcpyHostToDevice or hipMemcpyDeviceToHost).
hipLaunchKernelGGL: The macro used to launch a kernel with specific grid and block dimensions.
blockDim.x, blockIdx.x, threadIdx.x: Built-in variables used within a kernel to determine the unique ID of the current thread.
hipFree: Frees allocated device memory. 
GitHub
GitHub
 +4
Compilation
You can compile this code using the hipcc compiler provided by the AMD ROCm platform: 
AMD ROCm documentation
AMD ROCm documentation
 +1
bash
hipcc -o vector_square vector_square.cpp
./vector_square

***************************
**************************************
OpenCL Guide
<img width="1100" height="500" alt="image" src="https://github.com/user-attachments/assets/0fded6d9-495b-42e3-938f-4f19f3b346cb" />
What is OpenCL?
OpenCL (Open Computing Language) is an open, royalty-free standard for cross-platform, parallel programming of diverse accelerators found in supercomputers, cloud servers, personal computers, mobile devices and embedded platforms. OpenCL was created and is managed by the Khronos Group. OpenCL makes applications run faster and smoother by enabling them to use the parallel processing power in a system or device.

OpenCL is widely used throughout the industry. Many silicon vendors ship OpenCL with their processors, including GPUs, DSPs and FPGAs. The OpenCL API specification enables each chip to have to have its OpenCL drivers tuned to its specific architecture. Having the same, standardized API available on many systems enable developers to widely deploy their applications to reach more customers while minimizing porting and support costs.

To say they support conformant OpenCL, hardware vendors must become OpenCL Adopters and submit their conformance test results for review by the OpenCL Working Group at Khronos. You can view a list of hardware vendors with Conformant OpenCL Implementations on Khronos' OpenCL Adopter Page.
<img width="1807" height="909" alt="image" src="https://github.com/user-attachments/assets/5a2cd270-072f-4c47-a662-2ce7eea84fc4" />




OpenCL is widely deployed and used throughout the industry


By enabling applications to tap into the power of accelerated parallel programming, OpenCL greatly improves the speed and responsiveness of a wide range of applications, engines and libraries - including professional creative tools, scientific and medical software, vision processing, and neural network training and inferencing.

As well as being programmed directly by developers, OpenCL is also increasingly being used as a backend target for languages, compilers and machine learning stacks that need a portable API to reach into hardware acceleration for generated code.

You can find a list of applications that use OpenCL on Wikipedia.

********************************
How Does OpenCL Work?
OpenCL is a programming framework and runtime that enables a programmer to create small programs, called kernel programs (or kernels), that can be compiled and executed, in parallel, across any processors in a system. The processors can be any mix of different types, including CPUs, GPUs, DSPs, FPGAs or Tensor Processors - which is why OpenCL is often called a solution for heterogeneous parallel programming.

The OpenCL framework contains two APIs. The Platform Layer API is run on the host CPU and is used first to enable a program to discover what parallel processors or compute devices are available in a system. By querying for what compute devices are available an application can run portably across diverse systems - adapting to different combinations of accelerator hardware. Once the compute devices are discovered, the Platform API enables the application to select and initialize the devices it wants to use.

The second API is the Runtime API which enables the application's kernel programs to be compiled for the compute devices on which they are going to run, loaded in parallel onto those processors and executed. Once the kernel programs finish execution the Runtime API is used to gather the results.
<img width="984" height="986" alt="image" src="https://github.com/user-attachments/assets/82c26ee2-deb8-4e10-9662-29b27eb14922" />

******************
Compiling and Executing OpenCL Kernels


The most commonly used language for programming the kernels that are compiled and executed across the available parallel processors is called OpenCL C. OpenCL C is based on C99 and is defined as part of the OpenCL specification. Kernels written in other programming languages may be executed using OpenCL by compiling to an intermediate program representation, such as SPIR-V.

OpenCL is a low-level programming framework so the programmer has direct, explicit control over where and when kernels are run, how the memory they use is allocated and how the compute devices and host CPU synchronize their operations to ensure that data and computed results flow correctly - even when the host and compute kernels are running in parallel.

Executing an OpenCL Program
OpenCL regards a kernel program as the basic unit of executable code (similar to a C function). Kernels can execute with data or task-parallelism. An OpenCL program is a collection of kernels and functions (similar to dynamic library with run-time linking).

An OpenCL command queue is used by the host application to send kernels and data transfer functions to a device for execution. By enqueueing commands into a command queue, kernels and data transfer functions may execute asynchronously and in parallel with application host code.

The kernels and functions in a command queue can be executed in-order or out-of-order. A compute device may have multiple command queues.
<img width="1148" height="940" alt="image" src="https://github.com/user-attachments/assets/a4e82dbf-3ece-41cf-9dbe-ac06dc3b36f9" />

*******************
A complete sequence for executing an OpenCL program is:

Query for available OpenCL platforms and devices

Create a context for one or more OpenCL devices in a platform

Create and build programs for OpenCL devices in the context

Select kernels to execute from the programs

Create memory objects for kernels to operate on

Create command queues to execute commands on an OpenCL device

Enqueue data transfer commands into the memory objects, if needed

Enqueue kernels into the command queue for execution

Enqueue commands to transfer data back to the host, if needed


**************************************
Programming OpenCL Kernels
An OpenCL application is split into host code and device kernel code. Host code is typically written using a general programming language such as C or C++ and compiled by a conventional compiler for execution on the host CPU. OpenCL bindings for other languages are also available, such as Python.

Device kernels that are written in OpenCL C, which is based on C99, can be ingested and compiled by the OpenCL driver during execution of an application using runtime OpenCL API calls. This is called online compilation and is supported by all OpenCL drivers. OpenCL C is a subset of ISO C99 with language extensions for parallelism, well-defined numerical accuracy (IEEE 754 rounding with specified max error) and a rich set of built-in functions including cross, dot, sin, cos, pow, log etc.
<img width="1834" height="848" alt="image" src="https://github.com/user-attachments/assets/76d8ad0f-4273-41a9-bb1f-ed888b25b78b" />

The OpenCL specification also enables optional offline compilation where the kernel program is pre-compiled into a binary format that a particular driver can ingest. Offline compilation can add significant value to developers by:

Speeding OpenCL application execution by eliminating or minimizing kernel code compilation time.
Leveraging alternative kernel languages and tools to produce executable binaries.
There are two offline compilation approaches:

Kernels that are compiled online by the driver can be retrieved by the application using the clGetProgramInfo call. Those cached, device-specific, kernels can then be later reloaded for execution on the same device instead of re-compiling those kernels from the source code.
Offline compilers can be invoked independently before the OpenCL application executes to generate binaries to load and run on the device during application execution.
<img width="926" height="527" alt="image" src="https://github.com/user-attachments/assets/24c4d527-2a56-4de2-b6e7-879e3e412d1c" />


<img width="2018" height="835" alt="image" src="https://github.com/user-attachments/assets/64330e27-0928-4cef-8037-e82edebe5123" />


****************************************
OpenCL Programming Model
To understand how to program OpenCL in more detail let's consider the Platform, Execution and Memory Models. The three models interact and define OpenCL's essential operation.

Platform Model
The OpenCL Platform Model describes how OpenCL understands the compute resources in a system to be topologically connected.

A host is connected to one or more OpenCL compute devices. Each compute device is collection of one or more compute units where each compute unit is composed of one or more processing elements. Processing elements execute code with SIMD (Single Instruction Multiple Data) or SPMD (Single Program Multiple Data) parallelism.
<img width="1201" height="654" alt="image" src="https://github.com/user-attachments/assets/fd32346d-1eff-4c52-ac1d-573d5eb08ce9" />

Execution Model
OpenCL's clEnqueueNDRangeKernel command enables a single kernel program to be initiated to operate in parallel across an N-dimensional data structure. Using a two-dimensional image as a example, the size of the image would be the NDRange, and each pixel is called a work-item that a copy of kernel running on a single processing element will operate on.

As we saw in the Platform Model section above, it is common for processors to group processing elements into compute units for execution efficiency. Therefore, when using the clEnqueueNDRangeKernel command, the program specifies a work-group size that represents groups of individual work-items in an NDRange that can be accommodated on a compute unit. Work-items in the same work-group are able to share local memory, synchronize more easily using work-group barriers, and cooperate more efficiently using work-group functions such as async_work_group_copy that are not available between work-items in separate work-groups.

<img width="1058" height="910" alt="image" src="https://github.com/user-attachments/assets/1ec02115-0d13-4348-b91a-ca04ab440d51" />

Memory Model
OpenCL has a hierarchy of memory types:

Host memory - available to the host CPU

Global/Constant memory - available to all compute units in a compute device

Local memory - available to all the processing elements in a compute unit

Private memory - available to a single processing element
<img width="1128" height="984" alt="image" src="https://github.com/user-attachments/assets/0987e508-94a5-4e8a-a84f-725a50d767d0" />

********************
C++ for OpenCL
The OpenCL working group has transitioned from the original OpenCL C++ kernel language first defined in OpenCL 2.2 to the community developed C++ for OpenCL kernel language that provides improved features and compatibility with OpenCL C.

C++ for OpenCL enables developers to use most C++ features in kernel code while keeping familiar OpenCL constructs, syntax, and semantics from OpenCL C. This facilitates a smooth transition to new C++ features in existing OpenCL applications and does not require changing familiar development flows or tools. The main design goal of C++ for OpenCL is to reapply OpenCL-specific concepts to C++ in the same way as OpenCL C applies them to C. Aside from minor exceptions OpenCL C is a valid subset of C++ for OpenCL. Overall, kernel code written in C++ for OpenCL looks just like code written in OpenCL C with some extra C++ features available for convenience. C++ for OpenCL supports features from C++17.


<img width="964" height="835" alt="image" src="https://github.com/user-attachments/assets/4349533b-bf0a-4a77-a76d-323db005d2b7" />


**********************
Experimental support for C++ for OpenCL was added in Clang 9 with bug fixes and improvements in Clang 10.

You can check out C++ for OpenCL in Compiler Explorer.

Documentation
The language documentation can be found in releases of OpenCL-Docs with the first official version 1.0 (see also the latest WIP version html or pdf).

This documentation provides details about the language semantics as well as differences to OpenCL C and C++.

Example
The following code snippet illustrates how to implement kernels with complex number arithmetic using C++ features.

// This example demonstrates a convenient way to implement
// kernel code with complex number arithmetic using various
// C++ features.

// Define a class Complex, that can perform complex number
// computations with various precision when different
// types for T are used - double, float, half...
template<typename T>
class complex_t {
T m_re; // Real component.
T m_im; // Imaginary component.

public:
complex_t(T re, T im): m_re{re}, m_im{im} {};
complex_t operator*(const complex_t &other) const
{
  return {m_re * other.m_re - m_im * other.m_im,
           m_re * other.m_im + m_im * other.m_re};
}
int get_re() const { return m_re; }
int get_im() const { return m_im; }
};

// A helper function to compute multiplication over
// complex numbers read from the input buffer and
// to store the computed result into the output buffer.
template<typename T>
void compute_helper(global T *in, global T *out) {
  auto idx = get_global_id(0);	
  // Every work-item uses 4 consecutive items from the input
  // buffer - two for each complex number.
  auto offset = idx * 4;
  auto num1 = complex_t{in[offset], in[offset + 1]};
  auto num2 = complex_t{in[offset + 2], in[offset + 3]};
  // Perform complex number multiplication.
  auto res = num1 * num2;
  // Every work-item writes 2 consecutive items to the output
  // buffer.
  out[idx * 2] = res.get_re();
  out[idx * 2 + 1] = res.get_im();
}

// This kernel can be used for complex number multiplication
// in single precision.
kernel void compute_sp(global float *in, global float *out) {
  compute_helper(in, out);
}

// This kernel can be used for complex number multiplication
// in half precision.
#pragma OPENCL EXTENSION cl_khr_fp16: enable
kernel void compute_hp(global half *in, global half *out) {
  compute_helper(in, out); 
}
Developing kernels with C++ for OpenCL
C++ for OpenCL sources can be developed and compiled just like OpenCL C sources. However due to the increased growth of application complexity especially those that benefit most from a rich variety of C++ features and high-level abstractions, it is expected that the majority of C++ for OpenCL kernels will be compiled offline. Offline compilation also provides the ability to take advantage of various newest features in open source tools and frameworks without waiting for their integration into the vendor toolchains.

Offline compilation
Clang provides support for C++ for OpenCL using the same interface as for OpenCL C.

clang -cl-std=CLC++ test.cl
clang test.clcpp
More details can be found in its UsersManual. In the majority of cases the generated binary can be used in existing drivers. C++ for OpenCL version 1.0 is developed against OpenCL 2.0. Depending on the features used, drivers from other versions (e.g. OpenCL 3.0) might be able to load the binaries produced with C++ for OpenCL v1.0 too. Use of global objects and static function objects with non-trivial constructors is not supported in a portable way, refer to the following clang documentation for details.

Clang only supports a limited number of vendors and therefore to allow executing the binaries from C++ for OpenCL on more devices it is recommended to generate portable executable formats. C++ for OpenCL kernel sources can be compiled into SPIR-V using open source tools and then loaded into drivers supporting SPIR-V. C++ for OpenCL 1.0 mainly requires SPIR-V 1.0 plus SPIR-V 1.2 for some features.

The bugs and implementation of new features in clang can be tracked via the OpenCL Support Page.

Online compilation
Kernels written in C++ for OpenCL can be compiled online on devices that support the cl_ext_cxx_for_opencl extension.

Libraries
All OpenCL C builtin library functions are available with C++ for OpenCL.

There is work ongoing to provide C++ based libraries extended from the C++ implementation: see the experimental libraries in clang and libclcxx projects for more details.

Extensions
All extensions from OpenCL C are available with C++ for OpenCL.

Contributions
C++ for OpenCL is a community driven open language and contributions are welcome from anyone interested to improve the language compilation in clang or documentation of the language hosted in OpenCL-Docs. Refer to git log or git blame to find relevant contributors to contact or loop in for reviews.

***********************
Offline Compilation of OpenCL Kernel Sources
Aside from online compilation during application execution, OpenCL kernel sources can be compiled offline into binaries that can be loaded into the drivers using special API calls (e.g. clCreateProgramWithBinary or clCreateProgramWithIL).

This section describes available open source tools for offline compilation of OpenCL kernels.

Open Source Tools
clang is a compiler front-end for the C/C++ family of languages, including OpenCL C and C++ for OpenCL. It can produce executable binaries (e.g. AMDGPU), or portable binaries (e.g. SPIR). It is part of the LLVM compiler infrastructure project, and there is information regarding OpenCL kernel language support and standard headers.
SPIRV-LLVM Translator provides a library and the llvm-spirv tool for bidirectional translation between LLVM IR and SPIR-V.
clspv compiler and clvk runtime layer enable OpenCL applications to be executed with Vulkan drivers.
SPIR-V Tools provide a set of utilities to process SPIR-V binaries including spirv-opt optimizer, spirv-link linker, spirv-dis/spirv-as (dis-)assembler, and spirv-val validator.
Compiling Kernels to SPIR-V
The open source tools can be used to perform full compilation from OpenCL kernel sources into SPIR-V.


<img width="879" height="958" alt="image" src="https://github.com/user-attachments/assets/4aa7a5f9-ff03-4b71-aa76-686ee9e10b95" />

**************************************
*****************************
GPU Architectures and Programming
https://onlinecourses.nptel.ac.in/noc20_cs41/preview
**************************************************
By Prof. Soumyajit Dey   |   IIT Kharagpur
The course covers basics of conventional CPU architectures, their extensions for single instruction multiple data processing (SIMD) and finally the generalization of this concept in the form of single instruction multiple thread processing (SIMT) as is done in modern GPUs. We cover GPU architecture basics in terms of functional units and then dive into the popular CUDA programming model commonly used for GPU programming. In this context, architecture specific details like memory access coalescing, shared memory usage, GPU thread scheduling etc which primarily effect program performance are also covered in detail. We next switch to a different SIMD programming language called OpenCL which can be used for programming both CPUs and GPUs in a generic manner. Throughout the course we provide different architecture-aware optimization techniques relevant to both CUDA and OpenCL. Finally, we provide the students with detail application development examples in two well-known GPU computing scenarios.

INTENDED AUDIENCE  : Computer Science, Electronics, Electrical Engg students
PREREQUISITES  		: Programming and Data Structure, Digital Logic, Computer architecture
INDUSTRY SUPPORT	: NVIDIA, AMD, Google, Amazon and most big-data companies
Summary
Course Status :	Completed
Course Type :	Elective
Language for course content :	English
Duration :	12 weeks
Category :	
Computer Science and Engineering
Credit Points :	3
Level :	Undergraduate/Postgraduate
Start Date :	27 Jan 2020
End Date :	17 Apr 2020
Enrollment Ends :	03 Feb 2020
Exam Date :	26 Apr 2020 IST
Note: This exam date is subject to change based on seat availability. You can check final exam date on your hall ticket.

FacebookXTeamsLinkedInWhatsAppShare


Course layout
Week 1 :Review of Traditional Computer Architecture – Basic five stage RISC Pipeline, Cache Memory, Register File, SIMD instructions
Week 2 :GPU architectures - Streaming Multi Processors, Cache Hierarchy,The Graphics Pipeline
Week 3 :Introduction to CUDA programming
Week 4 :Multi-dimensional mapping of dataspace, Synchronization
Week 5 :Warp Scheduling, Divergence
Week 6 :Memory Access Coalescing
Week 7 :Optimization examples : optimizing Reduction Kernels
Week 8 :Optimization examples : Kernel Fusion, Thread and Block
Week 9 :OpenCL basics
Week 10 :OpenCL for Heterogeneous Computing
Week 11-12 :Application Design : Efficient Neural Network Training/Inferencing
Books and references
1. “Computer Architecture -- A Quantitative Approach” - John L.Hennessy and David A. Patterson
2. "Programming Massively Parallel Processors" - David Kirk and Wen-mei Hwu
3. Heterogeneous Computing with OpenCL” -- Benedict Gaster,Lee Howes, David R. Kaeli

*****************************************
***********************
