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
