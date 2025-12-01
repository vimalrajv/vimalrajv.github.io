# The Unspoken Parallels: Why LLM Pre-training and HPC Simulations Speak the Same Language
When I moved from Oil & Gas simulations into the current phase of my career, I kept noticing something that felt like déjà vu. Not in a mystical way, but in the deeply technical sense. The problems we're solving to make LLMs train faster—the inefficiencies we're fighting, the architectural decisions being celebrated as innovations—weren't exactly new. They were the same problems HPC engineers have been wrestling with for decades, just wearing different names and running on different hardware.​

The more I thought about it, the more it became clear that LLM pre-training isn't some radical departure from HPC. It's actually HPC evolution wearing a transformer hat. And honestly? We need to talk about the terminology.

### A Quick Note on What We're Calling Things Now
Here's the thing that gets me: the AI community has this habit of renaming concepts that HPC has been using for literally 20+ years and presenting them as novel innovations. I don't say this to be dismissive—I say it because recognizing these parallels could save teams years of reinventing wheels.

"Data parallelism" in ML? We've been doing this since the early days of MPI. You replicate the model (or simulation code) across nodes, split your data (or computational domain), process in parallel, and synchronize results. That's been standard practice in CFD simulations since the 1990s.​

"Tensor parallelism"? That's just distributed sparse matrix operations. When you split weight matrices across GPUs, you're doing exactly what we did when we partitioned finite element stiffness matrices across compute nodes in structural mechanics codes. Same principle, different application.​

"Pipeline parallelism"? HPC has been pipelining computational stages across nodes for decades. The specific implementation for transformers is new, but the concept of breaking sequential work into stages and overlapping execution across different processors is foundational HPC.​

"Gradient checkpointing" (or "activation checkpointing") is being celebrated as a memory breakthrough in ML. We called it checkpoint/restart in HPC, and we've been doing it since supercomputers had tape drives. The trade-off—recompute instead of store—is identical. We even have the same mathematical frameworks for deciding optimal checkpoint frequency.​

"Gradient accumulation"? That's domain decomposition with delayed boundary exchange. You solve local problems, accumulate results, synchronize periodically. We've been doing this in parallel PDE solvers forever.​

"Mixed precision training"? HPC numerical analysts have been managing precision hierarchies (float64/float32/float16) for numerical stability since GPUs were invented. The specific application to neural networks is new, but the principle of "use lower precision where you can, higher precision where you must" is classic numerical computing.​

The list goes on. AllReduce? Standard MPI collective operation since 1994. Ring-AllReduce? We've been using ring topologies for collective communication in HPC clusters for years. Flash Attention's "tiling" and "kernel fusion"? That's cache-aware algorithm design and operation fusion—core HPC optimization techniques from the early 2000s.​

I'm not saying the ML implementations aren't sophisticated or valuable. They absolutely are. But when I hear someone describe ZeRO optimizer as revolutionary, I think: "We've been partitioning optimizer state and doing state sharding in parallel solvers for decades." The specific application to transformer training is new, but the underlying memory optimization strategy is not.​

### The Fundamental Problem: Both Are Solving the Same Equation
Let's start with the obvious. Whether you're simulating fluid dynamics around a wind turbine or training an LLM with 70 billion parameters, you're facing the same core constraint: the compute-to-communication ratio is broken. Your GPUs (or CPUs) can perform billions of floating-point operations per second, but moving data around the system? That's the real bottleneck.​

In HPC, we've known this for years. The Floating Point Rate divided by Memory Bandwidth is what we call the "balance ratio," and it's been steadily worsening as GPUs get faster but interconnects don't. A modern GPU might have enough arithmetic throughput to solve a problem, but if it spends most of its time waiting for data to arrive from memory or from neighboring nodes, you're just spinning your wheels. Your 80% of peak performance becomes 20% of peak performance.​

LLM training faces exactly the same physics. When you're training a transformer, you're not just doing computation—you're constantly shuffling massive tensors around. During the prefill phase (when you process the entire input sequence), you're doing matrix-matrix multiplications, which is compute-bound and loves parallelism. During the decode phase (generating tokens one at a time), you're doing matrix-vector multiplications, which are memory-bound and make the GPU sit idle, watching bandwidth.​

The solutions people are celebrating as cutting-edge in ML—fused kernels, overlapping computation with communication, mixed precision arithmetic—are the exact playbook HPC has been running for 15 years.​

### Where HPC Has Already Won: The Techniques LLMs Are "Discovering"
##### Activation Checkpointing Is Just Checkpoint/Restart
The pitch for gradient checkpointing in ML: instead of storing all intermediate activations during the forward pass (which consumes huge amounts of memory), you save only a subset and recompute the rest during the backward pass. This trades computation for memory—you're doing extra forward passes to avoid storing everything.​

In HPC, we've called this checkpoint/restart since the 1980s. The exact same trade-off: don't store everything, recompute what you need when you need it. We even developed sophisticated theoretical frameworks (like the Young/Daly formula) to determine optimal checkpoint intervals based on failure rates and checkpoint overhead.​

The ML community is now developing "selective activation checkpointing" where you choose which operations to recompute based on their computational cost. In HPC, we've been doing selective rematerialization for years—deciding which intermediate results to cache versus recompute based on arithmetic intensity and memory pressure. Same concept, same trade-offs.​

##### AllReduce and Ring-AllReduce: MPI Since the 1990s
When gradient synchronization happens in distributed LLM training, the standard approach is AllReduce—a collective communication operation where each worker contributes data, and all workers receive the combined result. For large models, people use ring-AllReduce to avoid bandwidth bottlenecks by passing data in a ring topology.​

These are literal MPI primitives from the 1990s. HPC codes have been using MPI_Allreduce for decades to synchronize simulation state across distributed nodes. The ring algorithm for AllReduce? Published in parallel computing textbooks 20 years ago. The recent "innovations" in ML are mostly about implementing these on GPU interconnects (NVLink, InfiniBand) instead of traditional HPC networks—but the algorithms are identical.​

##### Communication Hiding: The Art of Making the Network Disappear
One of the most critical optimizations in both domains is overlapping computation with communication. In MPI-based HPC codes, you send data asynchronously with MPI_Isend while continuing calculations, so the network bandwidth gets "hidden" behind compute time. In transformer training, frameworks like PyTorch use gradient bucketing and asynchronous AllReduce operations to achieve the same thing—gradient synchronization happens in the background while backward computation continues.​

The problem statement is identical: network latency and bandwidth are finite resources. If you can schedule your communication to happen when the GPU would be idle anyway, you've just purchased free performance. The implementation details differ, but the principle is pure HPC: respect the memory hierarchy and communication costs, and structure your algorithm accordingly.​

We've been doing this for years. The techniques are documented in every MPI optimization guide.​

##### Gradient Clipping Is Numerical Stability 101
The ML community discovered that clipping gradients—limiting their magnitude to prevent explosive updates—dramatically improves training stability, especially for models with steep loss landscapes. This is now standard practice.​

In HPC, we've been doing gradient clipping and norm-based stabilization in iterative solvers since forever. When you're solving nonlinear PDEs with Newton-Raphson or other iterative methods, controlling update magnitudes to prevent divergence is Numerical Methods 101. The specific application to neural network gradients is new, but the underlying stability principle—bound your updates to prevent numerical overflow—is ancient.​

##### ZeRO Optimizer: State Partitioning We've Done for Decades
The Zero Redundancy Optimizer (ZeRO) is celebrated as a breakthrough that partitions optimizer states, gradients, and parameters across GPUs to reduce memory redundancy. Instead of each GPU storing a full copy of everything, ZeRO partitions the state and reconstructs it on-demand via communication.​

In HPC, we've been doing state partitioning and distributed state management in parallel solvers for decades. When you're running a massive CFD simulation, you don't replicate the entire solution vector on every node—you partition it, each node owns a piece, and you exchange boundary data as needed. The mathematics differ (optimizer states vs. solution vectors), but the architectural pattern is identical: eliminate memory redundancy through partitioning, reconstruct via communication.​

ZeRO's three stages (partitioning optimizer states, gradients, then parameters) are clever implementations, but the principle is not new.​

##### Flash Attention: Cache-Aware Algorithm Design
Flash Attention is an optimized attention mechanism that uses tiling and kernel fusion to reduce memory reads/writes between GPU high-bandwidth memory (HBM) and on-chip SRAM. By keeping data in fast SRAM and fusing operations, it achieves dramatic speedups.​

HPC has been doing cache-aware algorithm design and kernel fusion since the early 2000s. Blocking/tiling algorithms to fit data in cache? That's been a cornerstone of numerical linear algebra libraries (BLAS, LAPACK) for decades. Fusing operations to reduce memory traffic? Standard practice in GPU computing since CUDA was invented.​

Flash Attention's specific application to transformer attention is excellent engineering, but the techniques—tiling, fusion, minimizing HBM access—are textbook HPC optimization.​

##### The Shared Enemy: The Communication-to-Compute Bottleneck
Both HPC and LLM training are fundamentally limited by the same physics: data motion is expensive, and it's getting more expensive relative to computation.​

In fluid dynamics simulations running on Cray supercomputers, the bottleneck is often interconnect bandwidth between nodes. In LLM training on GPU clusters, the bottleneck is often NVLink bandwidth between GPUs or network bandwidth between nodes. The specific hardware is different, but the fundamental constraint is identical.​

This is why both domains obsess over the same metrics:

Memory bandwidth efficiency: Can you saturate the available bandwidth, or are you leaving performance on the table?​

Communication volume: Can you reduce the amount of data you're moving?​

Synchronization overhead: Are workers waiting for each other unnecessarily?​

The HPC community solved similar problems by being ruthlessly pragmatic. If communication becomes too expensive, you change the algorithm. Pipeline parallelism in HPC (where different stages of computation happen on different nodes simultaneously) directly inspired pipeline parallelism in LLM training. Tensor parallelism—splitting weight matrices across GPUs—is conceptually similar to distributed sparse matrix operations in HPC.​

We've been doing this for years.

### What LLM Training Can Learn From HPC's Experience
The honest truth is that LLM training is catching up to techniques HPC has already mastered, and there's still significant room to learn.

Numerical stability and error tracking: HPC codes use sophisticated schemes to track numerical error accumulation over millions of iterations. In LLM training, we're still pretty naive about this—we watch loss curves and validation metrics, but we don't deeply understand how floating-point errors accumulate through a billion-step training run. HPC has better tools here.​

Performance profiling and bottleneck identification: HPC has amazing profiling tools (Intel VTune, NVIDIA Nsight, HPCToolkit) that give you detailed breakdowns of where time is spent—on computation, communication, I/O, synchronization. The ML community has some tools, but nothing as mature. If ML teams invested in HPC-grade profiling, they'd likely discover substantial performance headroom.​

I/O optimization: Managing data flow from storage to compute is a science in HPC. Parallel I/O libraries, data staging, and intelligent prefetching are well-established. LLM training pipelines often treat I/O casually, and I suspect there are significant wins to be had by taking it more seriously.​

Load balancing under variability: Modern supercomputers deal with dynamic voltage and frequency scaling (DVFS), heterogeneous hardware, and unpredictable node performance. There are sophisticated techniques for detecting load imbalances in real-time and rebalancing work dynamically. LLM training clusters could benefit from similar approaches.​

### The Bigger Picture: Why This Matters
The reason I'm writing this is not to diminish the genuine innovation in LLM training. But I think there's value in recognizing that a huge portion of LLM training optimization is not new science—it's applied HPC.​

This has practical implications:

1. Stop renaming things. Or at least acknowledge the lineage. When you call something "gradient checkpointing," mention that it's the same checkpoint/restart strategy HPC has used for 30 years. When you talk about AllReduce, acknowledge it's an MPI primitive. This helps cross-pollination and prevents teams from rediscovering known limitations.

2. Recruit HPC engineers to ML infrastructure teams. If you're building LLM training systems, hiring someone who spent a decade optimizing CFD codes or running simulations on Exascale supercomputers is not a sideways move. It's hiring someone who's already solved 80% of your problems—they just called them different names.

3. Use HPC literature. The collective communication literature in HPC is enormous. Research on optimal checkpoint intervals, load balancing, numerical stability, cache-aware algorithms—it's all directly applicable. Stop treating ML as a greenfield and start mining the 40 years of HPC research.

4. Invest in profiling and measurement. You can't optimize what you don't understand. LLM training teams would be better off with HPC-grade instrumentation and analysis tools rather than relying on training curves and wall-clock time.

5. Think about heterogeneous hardware earlier. HPC already deals with systems where not all GPUs are identical, where network topology matters, where power consumption is a first-class constraint. LLM training will too, and the HPC playbook is well-established.

### We've Done This For Years
Look, I get it. The ML community has its own culture, its own terminology, its own conferences. And there's genuine innovation happening in how these classical techniques are applied to transformers and large language models.

But let's be clear: when someone presents "data parallelism" as a new concept, we've been doing that in HPC since the 1990s. When "gradient checkpointing" is described as innovative, we've been checkpointing and recomputing for decades. When "ring-AllReduce" is touted as clever, it's literally in MPI textbooks from 20 years ago.​

The ultimate insight is simple but profound: LLM pre-training is a massive distributed numerical computing problem. It involves iterative refinement, communication overhead, numerical stability, fault tolerance, and resource allocation. That's not "AI engineering"—that's high-performance computing. And we've been doing it for decades.​

The differences are real (transformer operations are different from finite element kernels), but the similarities are deeper. Both are fundamentally about extracting maximum performance from heterogeneous systems with limited memory bandwidth and network capacity. Both require careful attention to numerical precision, synchronization, and load balancing. Both benefit from the same classical optimizations: better algorithms, better data layout, better communication overlap, and better understanding of your actual hardware constraints.

If you've spent time in HPC, LLM training isn't mysterious—it's recognizable. The terminology might be different, but the problems are the same. We've been solving communication bottlenecks, managing memory hierarchies, checkpointing for fault tolerance, and optimizing collective operations for years.

And if you're new to LLM training, understanding the HPC perspective might save you from optimizing things that don't matter while missing the real bottlenecks. Sometimes the most cutting-edge solution is just a well-executed version of something an HPC engineer figured out in 2005—we just called it something else.

We've done this for years. It's time the terminology caught up with the reality.