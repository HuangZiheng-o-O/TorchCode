# FlashAttention深度讲解

**Created:** 4/10/2026 21:02:48  
**Updated:** 4/10/2026 21:07:47  
**Exported:** 4/10/2026 21:08:25  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99db1-d618-83ea-bf3c-d223a9c4fb66](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99db1-d618-83ea-bf3c-d223a9c4fb66)  

## Prompt:
4/10/2026, 9:02:47 PM

FlashAttention深度讲解 包括面试所有可能问题 搜索 思考  think harder

## Response:

 **High Bandwidth Memory（高带宽存储器）**
 
**结论先说：FlashAttention 的本质不是“改写 Attention 数学”，而是“改写 Attention 的执行顺序与内存访问方式”。它仍然精确计算**  $\mathrm{softmax}(QK^\top)V$  **，并没有做近似；它真正优化的是把原本会写回 HBM 的巨大中间矩阵按块在 SRAM / shared memory 中完成，从而把瓶颈从“显存读写”转向“更高效的片上计算”。因此它的核心价值不是降低理论算术复杂度，而是降低 IO complexity、显著节省显存，并带来真实 wall-clock 加速。** [arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**The core answer is this: FlashAttention does not change the mathematics of attention; it changes the execution order and memory-access pattern. It still computes**  $\mathrm{softmax}(QK^\top)V$  **exactly, with no approximation. What it optimizes is avoiding writing the huge intermediate attention matrices back to HBM by processing them block by block in SRAM / shared memory. So its main gain is not reducing arithmetic complexity, but reducing IO complexity, memory footprint, and real wall-clock time.** [arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

* * *

1\. 先把标准 Attention 讲清楚
----------------------

**标准 scaled dot-product attention 的公式是：**

$$
S=\frac{QK^\top}{\sqrt{d}},\qquad P=\mathrm{softmax}(S),\qquad O=PV
$$

**中文解释：** 这里  $Q,K,V$  分别是 query、key、value， $d$  是 head dimension， $S$  是注意力分数矩阵， $P$  是按行 softmax 后的注意力概率， $O$  是输出。这个公式本身没有问题，问题在于工程实现通常会显式物化  $S$  和  $P$ ，而它们的形状都是  $N\times N$ 。当序列长度  $N$  很大时，显存访问和中间张量存储会成为真正瓶颈。这个瓶颈往往比“乘法本身”更致命。 [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**English explanation:** Here  $Q,K,V$  are the query, key, and value matrices,  $d$  is the head dimension,  $S$  is the attention score matrix,  $P$  is the row-wise softmax probability matrix, and  $O$  is the final output. The formula itself is not the problem. The systems problem is that standard implementations often materialize  $S$  and  $P$ , both of shape  $N\times N$ . For large sequence length  $N$ , memory traffic and intermediate storage dominate runtime. In practice, that bottleneck is often worse than the raw multiplications. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**面试里最容易加分的一句话是：标准 attention 的算术复杂度是二次的，但 FlashAttention 真正瞄准的是内存层级上的 IO 代价。** 原论文明确强调，很多近似 attention 方法虽然减少了某些计算，但未必带来 wall-clock speedup，因为它们没有从 GPU memory hierarchy 的角度优化 attention。FlashAttention 的关键思想就是 IO-aware。 [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**A high-scoring interview sentence is this: standard attention is quadratic in arithmetic, but FlashAttention mainly targets the IO cost across the GPU memory hierarchy.** The original paper explicitly argues that many approximate attention methods reduce some computation yet still fail to deliver wall-clock gains, because they do not optimize around the memory hierarchy. FlashAttention is fundamentally IO-aware. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

* * *

2\. FlashAttention 到底做了什么
-------------------------

**FlashAttention 做的事情可以概括成一句话：把  $Q,K,V$  分块，把 softmax 的统计量在线维护，把中间结果留在片上内存里，避免显式存整个  $N\times N$  attention matrix。** 论文中的 Algorithm 1 展示了这个过程：它分块读取  $K,V$ ，再遍历对应的  $Q$  block，在片上计算局部分数、局部 softmax 统计量和输出更新，最后只把必要的输出和归一化统计量写回 HBM。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**FlashAttention can be summarized in one sentence: tile  $Q,K,V$ , maintain softmax statistics online, and keep intermediates on-chip so the full  $N\times N$  attention matrix is never materialized.** Algorithm 1 in the paper shows exactly this procedure: it loads  $K,V$  blocks, iterates over the corresponding  $Q$  blocks, computes local scores, local softmax statistics, and output updates on-chip, and writes back only the necessary outputs and normalization statistics to HBM. [arXiv+1](https://arxiv.org/pdf/2205.14135)

**因此，FlashAttention 不是“少算了很多”，而是“少搬了很多”。** 它前向仍然是精确 attention，仍然有  $O(N^2d)$  FLOPs；但额外内存从标准实现里对  $N\times N$  中间矩阵的依赖，降成了只需  $O(N)$  级别的附加存储。原论文给出的结论就是：Algorithm 1 返回精确的  $O=\mathrm{softmax}(QK^\top)V$ ，需要  $O(N^2d)$  FLOPs，但只需要  $O(N)$  additional memory。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**So FlashAttention does not mainly “compute much less”; it mainly “moves much less data.”** The forward pass is still exact attention and still uses  $O(N^2d)$  FLOPs. But the extra memory drops from dependence on the explicit  $N\times N$  intermediates to only  $O(N)$  additional storage. The paper states this directly: Algorithm 1 returns the exact  $O=\mathrm{softmax}(QK^\top)V$ , uses  $O(N^2d)$  FLOPs, and requires only  $O(N)$  additional memory. [arXiv+1](https://arxiv.org/pdf/2205.14135)

* * *

3\. 为什么 online softmax 是关键
--------------------------

**FlashAttention 能分块做而不失真，核心在于 softmax 可以通过“在线统计量”正确合并。** 设当前已经处理过一部分 logits，对每一行维护运行中的最大值  $m$  和归一化分母  $\ell$ 。当新块到来时，不需要重新看到所有历史元素，只要更新这两个统计量，就能得到与一次性 softmax 完全一致的结果。论文给出的核心关系是：**拼接后的 softmax 可以由分块的最大值和指数和递推得到。** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**The reason FlashAttention can be blockwise without approximation is that softmax can be merged exactly through online statistics.** For each row, it maintains the running maximum  $m$  and normalization denominator  $\ell$  over the logits processed so far. When a new block arrives, it does not need all historical logits again; updating these statistics is sufficient to produce the exact same result as full softmax. The paper’s key observation is that the softmax over concatenated blocks can be reconstructed from blockwise maxima and sums of exponentials. [arXiv+1](https://arxiv.org/pdf/2205.14135)

**常用面试公式可以这样写：**

$$
m^{\text{new}}=\max(m,\tilde m)
$$
 
$$
\ell^{\text{new}}=e^{m-m^{\text{new}}}\ell + e^{\tilde m-m^{\text{new}}}\tilde \ell
$$
 
$$
O^{\text{new}}=\frac{e^{m-m^{\text{new}}}\,\ell\, O + e^{\tilde m-m^{\text{new}}}\,\tilde P V_{\text{block}}}{\ell^{\text{new}}}
$$

**中文解释：**  $m$  是当前已处理部分每一行的最大 logit， $\ell$  是对应的指数和， $\tilde m,\tilde\ell$  是新块的局部统计量， $O$  是当前累计输出。这个递推式的作用，是在数值稳定的前提下，把“旧块结果”和“新块结果”精确合并。它解释了为什么 FlashAttention 能够 block-by-block 做 softmax，同时保持 exactness。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**English explanation:**  $m$  is the row-wise maximum logit over the portion processed so far,  $\ell$  is the corresponding sum of exponentials,  $\tilde m,\tilde\ell$  are the local statistics of the new block, and  $O$  is the accumulated output. The role of these recurrences is to merge the “old block result” and the “new block result” exactly while preserving numerical stability. This is the mathematical reason FlashAttention can perform blockwise softmax without approximation. [arXiv+1](https://arxiv.org/pdf/2205.14135)

**这也是它数值稳定的原因。** 传统 softmax 要先减去行最大值避免溢出，FlashAttention 在块级别上维护同样的稳定化逻辑，本质上是把 log-sum-exp 的思想在线化。面试时你可以说：它不是绕开了 softmax 的稳定性问题，而是把稳定 softmax 的统计量变成了 streaming / online 形式。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**This is also why it is numerically stable.** Standard softmax subtracts the row maximum to avoid overflow. FlashAttention preserves that same stabilization logic at the block level; conceptually, it turns log-sum-exp into a streaming or online procedure. In an interview, a strong line is: it does not bypass softmax stability; it reformulates stable softmax into an online form. [arXiv+1](https://arxiv.org/pdf/2205.14135)

* * *

4\. 它为什么更快：真正原因是 IO complexity 更低
---------------------------------

**原论文最重要的理论结论之一是：标准 attention 需要**

$$
\Theta(Nd + N^2)
$$

**量级的 HBM accesses，而 FlashAttention 需要**

$$
\Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

**量级的 HBM accesses，其中  $M$  是片上 SRAM 大小。对于典型 head dimension 和 SRAM 容量，FlashAttention 的 HBM 访问次数明显更少。** [arXiv](https://arxiv.org/pdf/2205.14135)

**One of the most important theoretical results in the original paper is that standard attention requires**

$$
\Theta(Nd + N^2)
$$

**HBM accesses, while FlashAttention requires**

$$
\Theta\!\left(\frac{N^2 d^2}{M}\right)
$$

**HBM accesses, where  $M$  is the on-chip SRAM size. For typical head dimensions and SRAM sizes, FlashAttention performs far fewer HBM accesses.** [arXiv](https://arxiv.org/pdf/2205.14135)

**中文解释：** 这里  $N$  是序列长度， $d$  是每个 head 的维度， $M$  是片上快速存储容量。这个公式的意义不是说 FlashAttention 改变了 attention 的  $N^2$  本质，而是说它把“最慢层级上的数据搬运”显著压缩了。GPU 上很多 kernel 的真实性能不是由 FLOPs 决定，而是由 memory bandwidth 决定，所以减少 HBM 读写会直接转化成吞吐提升。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**English explanation:** Here  $N$  is sequence length,  $d$  is head dimension, and  $M$  is the on-chip fast-memory capacity. The meaning is not that FlashAttention changes the fundamental  $N^2$  nature of attention. Rather, it dramatically reduces data movement at the slowest memory level. On GPUs, many kernels are limited not by FLOPs but by memory bandwidth, so reducing HBM traffic directly improves throughput. [arXiv+1](https://arxiv.org/pdf/2205.14135)

**所以最标准的面试回答是：FlashAttention 优化的是 memory-bound 部分，而不是把 quadratic attention 变成 linear attention。** 这句话非常重要，因为很多候选人会误答成“FlashAttention 把时间复杂度从  $O(N^2)$  降到  $O(N)$ ”，这是错误的。它降低的是额外显存和 IO 成本，不是把精确 attention 的算术复杂度改成线性。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**So the most standard interview answer is: FlashAttention optimizes the memory-bound part of attention, not by turning quadratic attention into linear attention.** This sentence matters because a common mistake is to claim that FlashAttention reduces exact attention from  $O(N^2)$  time to  $O(N)$ . That is false. It reduces extra memory use and IO cost; it does not make exact dense attention arithmetically linear. [arXiv+1](https://arxiv.org/pdf/2205.14135)

* * *

5\. 前向和反向为什么都受益
---------------

**前向加速容易理解，反向更值得讲。** 传统反向传播通常需要保存  $S$  或  $P$  这类大中间张量，显存很重。FlashAttention 的策略是：前向只保存足够小的归一化统计量，例如 log-sum-exp 相关量；到反向时，再从  $Q,K,V$  block 级重算局部 attention。论文把这称为 recomputation，并指出这是一种 selective gradient checkpointing：多做一点算术，换更少的 HBM 读写和更低的显存占用。 [arXiv+1](https://arxiv.org/pdf/2205.14135)

**The forward-pass gain is intuitive, but the backward pass is where the explanation becomes stronger.** Standard backprop often stores large intermediates such as  $S$  or  $P$ , which is memory expensive. FlashAttention instead stores only small normalization statistics, such as log-sum-exp related values, and then recomputes local attention blocks from  $Q,K,V$  during backward. The paper calls this recomputation and describes it as a form of selective gradient checkpointing: a bit more arithmetic in exchange for less HBM traffic and much lower memory usage. [arXiv+1](https://arxiv.org/pdf/2205.14135)

**这里的 trade-off 是典型面试加分点：FlashAttention 用“重算”换“省内存与省 IO”。** 这不是白送的优化，因为你确实做了额外计算；但 GPU 上常见情况是，额外 matmul 计算比反复读写大矩阵更便宜，所以总 wall-clock 反而更快。原论文报告，反向也能获得明显 speedup 和显著内存节省。 [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**The key trade-off, and a strong interview point, is that FlashAttention trades recomputation for lower memory and IO.** This is not a free optimization, because you do perform extra computation. But on GPUs, additional matrix math is often cheaper than repeatedly reading and writing large matrices, so total wall-clock time still improves. The papers report clear backward-pass speedups and substantial memory savings. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

* * *

6\. FlashAttention-2 比 FlashAttention-1 强在哪里
--------------------------------------------

**FlashAttention-2 的核心不是再提出一个新数学，而是把 GPU 并行划分做得更好。** 论文明确指出，FlashAttention-1 虽然已经显著提速，但离优化后的 GEMM 还很远，只达到理论峰值 FLOPs 的 25% 到 40% 左右。FlashAttention-2 的改进集中在三点：减少 non-matmul FLOPs、把单个 head 的计算沿序列维进一步并行到不同 thread blocks、并在 block 内重新分配 warp 工作以减少 shared memory 通信。 [arXiv+1](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

**The key idea of FlashAttention-2 is not new math, but better GPU work partitioning.** The paper states that FlashAttention-1, while already much faster, still remains far from optimized GEMM kernels, achieving only around 25% to 40% of peak FLOPs. FlashAttention-2 improves this along three dimensions: reducing non-matmul FLOPs, parallelizing even a single head across the sequence dimension into multiple thread blocks, and redistributing work within a block across warps to reduce shared-memory communication. [arXiv+1](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

**为什么要强调 non-matmul FLOPs？因为在 A100 这类 GPU 上，matmul 由 Tensor Cores 加速得非常快，而非矩阵乘操作便宜程度远不如 matmul。** FlashAttention-2 论文给出的例子是：A100 上 FP16/BF16 matmul 的理论吞吐远高于非 matmul FP32，所以从系统视角看，“每 1 个 non-matmul FLOP”比“每 1 个 matmul FLOP”贵得多。结论就是：要尽量把时间花在 Tensor Core 擅长的事上。 [arXiv+1](https://arxiv.org/pdf/2307.08691)

**Why emphasize non-matmul FLOPs? Because on GPUs such as A100, matmuls are heavily accelerated by Tensor Cores, while non-matmul operations are comparatively much more expensive.** FlashAttention-2 gives the example that the theoretical FP16/BF16 matmul throughput on A100 is far higher than the non-matmul FP32 throughput. From a systems perspective, each non-matmul FLOP is much more expensive than each matmul FLOP. The practical conclusion is to spend as much time as possible doing Tensor-Core-friendly work. [arXiv+1](https://arxiv.org/pdf/2307.08691)

**FlashAttention-2 还做了一个很适合面试讲的工程优化：它不再保留前向里所有中间统计量，而是仅保存 backward 真正需要的 logsumexp  $L$ 。** 论文明确写到，没必要同时保存 row max 和 exp-sum，只需要

$$
L^{(j)} = m^{(j)} + \log \ell^{(j)}
$$

**即可。**

**中文解释：** 这里  $m^{(j)}$  是到第  $j$  个 block 为止的行最大值， $\ell^{(j)}$  是稳定 softmax 的归一化和， $L^{(j)}$  是 logsumexp。保存  $L$  的作用，是让反向足够恢复需要的 softmax 归一化信息，同时进一步减少存储开销。 [arXiv+1](https://arxiv.org/pdf/2307.08691)

**FlashAttention-2 also includes an engineering optimization that is excellent to mention in interviews: it stores only the logsumexp  $L$  actually needed by backward, instead of keeping all forward statistics.** The paper explicitly notes that it is unnecessary to store both the row max and the sum of exponentials; it is enough to store

$$
L^{(j)} = m^{(j)} + \log \ell^{(j)}
$$

**only.**

**English explanation:** Here  $m^{(j)}$  is the row-wise maximum up to block  $j$ ,  $\ell^{(j)}$  is the stable-softmax normalization sum, and  $L^{(j)}$  is the logsumexp. The role of storing  $L$  is to preserve exactly the normalization information backward needs, while further reducing storage overhead. [arXiv+1](https://arxiv.org/pdf/2307.08691)

**一句话总结 FA2：FA1 解决“不要把大矩阵写回 HBM”，FA2 进一步解决“怎样把 GPU 算力吃满”。** 论文报告，FA2 相比 FA1 大约再快 2 倍，并在 A100 上达到 50% 到 73% 的理论最大 FLOPs。 [arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

**One-line summary of FA2: FA1 solves “do not write the huge matrix back to HBM,” and FA2 further solves “how to saturate the GPU better.”** The paper reports roughly a further 2x speedup over FA1 and 50% to 73% of theoretical peak FLOPs on A100. [arXiv](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

* * *

7\. FlashAttention-3 又进化了什么
---------------------------

**FlashAttention-3 的定位非常明确：它主要是为 Hopper GPU 优化的版本。** 官方仓库写得很清楚，FA3 beta 主要针对 Hopper，如 H100 / H800；支持 FP16 / BF16 forward+backward，以及 FP8 forward，并要求较新的 CUDA 版本。 [GitHub](https://github.com/dao-ailab/flash-attention)

**FlashAttention-3 has a very clear positioning: it is primarily optimized for Hopper GPUs.** The official repository states that the FA3 beta targets Hopper GPUs such as H100 / H800; it supports FP16 / BF16 forward and backward plus FP8 forward, and requires a recent CUDA version. [GitHub](https://github.com/dao-ailab/flash-attention)

**它的核心提升点有三类：利用 Hopper 的异步能力把计算与搬运重叠；把 block-wise matmul 和 softmax 进一步流水化；以及在 FP8 下通过 block quantization 和 incoherent processing 降低量化误差。** Tri Dao 的官方博客和论文都强调了这些点。其本质是：FA3 不只是 tile 得更好，而是开始利用新一代硬件指令与异步执行模型。 [Tri Dao+1](https://tridao.me/blog/2024/flash3/)

**Its main improvements fall into three categories: exploiting Hopper’s asynchrony to overlap compute and data movement, further pipelining blockwise matmul and softmax, and reducing FP8 quantization error via block quantization and incoherent processing.** Both Tri Dao’s official blog and the paper emphasize these points. The essence is that FA3 is not merely better tiling; it starts to fully exploit the new hardware instructions and asynchronous execution model. [Tri Dao+1](https://tridao.me/blog/2024/flash3/)

**为什么 Hopper 上需要 FA3？因为 FA2 在 H100 上仍明显没有吃满硬件。** FA3 论文指出，FA2 在 H100 上的利用率明显低于优化好的 GEMM；博客进一步给出结论：FA3 在 H100 上相对 FA2 的 FP16 速度大约再提升 1.5 到 2 倍，可达到约 740 TFLOPS，约为 H100 理论峰值的 75%；FP8 则接近 1.2 PFLOPS。 [arXiv+1](https://arxiv.org/pdf/2407.08608)

**Why is FA3 needed on Hopper? Because FA2 still leaves substantial performance on the table on H100.** The FA3 paper states that FA2’s utilization on H100 remains well below optimized GEMM kernels; the blog further reports that FA3 is about 1.5x to 2x faster than FA2 in FP16 on H100, reaching around 740 TFLOPS, roughly 75% of H100 theoretical peak, while FP8 approaches 1.2 PFLOPS. [arXiv+1](https://arxiv.org/pdf/2407.08608)

**FA3 还有一个很强的面试亮点：它不只是更快，而且在数值上并不比标准实现更差。** 论文报告，FP16 下 FA3 与 FA2 的数值误差相同，并且都优于标准 attention 实现，因为中间 softmax rescaling 保持在 FP32；在 FP8 下，借助 block quantization 和 incoherent processing，FA3 相比基线 FP8 attention 的误差降低了 2.6 倍。 [arXiv](https://arxiv.org/pdf/2407.08608)

**FA3 also has a strong interview angle: it is not only faster, but can also be numerically better than a standard implementation.** The paper reports that in FP16, FA3 has the same numerical error as FA2 and both outperform the standard attention implementation because intermediate softmax rescaling is kept in FP32; in FP8, with block quantization and incoherent processing, FA3 reduces error by 2.6x versus the baseline FP8 attention. [arXiv](https://arxiv.org/pdf/2407.08608)

* * *

8\. 它和“近似 Attention”有什么根本区别
---------------------------

**FlashAttention 与 Performer、Linformer、稀疏 attention 这类方法的根本区别是：FlashAttention 是 exact attention 的高效实现，而不是 attention 近似。** 它没有改变输出定义，仍然精确返回

$$
O=\mathrm{softmax}(QK^\top)V
$$

因此它一般不会引入“近似误差换速度”的建模风险；它换的是“更合理的 kernel 设计和内存路径”。 [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**The fundamental difference between FlashAttention and methods such as Performer, Linformer, or sparse attention is that FlashAttention is an efficient implementation of exact attention, not an approximation.** It does not alter the output definition and still returns

$$
O=\mathrm{softmax}(QK^\top)V
$$

exactly. So it usually avoids the modeling trade-off of “approximation error for speed”; instead, it trades on better kernel design and memory movement. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**这意味着什么？意味着如果你的模型质量已经依赖 dense exact attention，FlashAttention 往往是“几乎没有算法风险”的优化。** 当然，它仍有硬件和实现前提，比如 kernel 支持、head dimension、mask 形式、dtype、GPU 架构等，但从模型语义上，它不改变 attention 的定义。 [GitHub+1](https://github.com/dao-ailab/flash-attention)

**What does that imply? It means that if your model quality already depends on dense exact attention, FlashAttention is often an optimization with almost no algorithmic risk.** Of course, it still has hardware and implementation constraints, such as kernel support, head dimension, mask form, dtype, and GPU architecture, but semantically it does not change the definition of attention. [GitHub+1](https://github.com/dao-ailab/flash-attention)

* * *

9\. 它的限制和边界条件，面试一定会问
--------------------

**第一，FlashAttention 并没有把 dense exact attention 变成线性时间。** 当  $N$  极大时，算术上依然有  $N^2$  级别的  $QK^\top$  交互，所以它不是解决超长上下文的唯一答案；它是把 dense attention 在现有 GPU 上做到更接近硬件上限。 [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**First, FlashAttention does not make dense exact attention linear-time.** For very large  $N$ , the arithmetic still involves  $N^2$ \-scale  $QK^\top$  interactions, so it is not the only answer to extreme long-context scaling; it is the way to make dense attention much closer to hardware-efficient on current GPUs. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**第二，它是强硬件相关优化。** 官方仓库明确列出支持的 GPU 架构、dtype 和 head dimension 范围，例如当前主仓库支持 Ampere / Ada / Hopper，常见场景下支持到 head dimension 256；FA3 beta 主要针对 Hopper，并对 CUDA 版本有要求。也就是说，FlashAttention 的收益强依赖你的硬件代际。 [GitHub](https://github.com/dao-ailab/flash-attention)

**Second, it is strongly hardware-dependent.** The official repository explicitly lists supported GPU architectures, dtypes, and head-dimension ranges; for example, the main repo supports Ampere / Ada / Hopper and commonly supports head dimensions up to 256, while the FA3 beta primarily targets Hopper and requires newer CUDA versions. In other words, FlashAttention’s gains depend heavily on your hardware generation. [GitHub](https://github.com/dao-ailab/flash-attention)

**第三，它的收益并不是在所有场景都同样大。** 当序列很短、batch 很小、或者框架本身已经把常规 attention kernel 优化得很好时，FlashAttention 的绝对收益可能没有长序列训练时那么惊艳。它最擅长的是长序列、内存压力大、HBM 带宽成为瓶颈的场景。这个判断与原论文和后续版本论文的实验结论一致。 [arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**Third, its gains are not equally large in every scenario.** When sequences are short, batches are tiny, or the framework already has highly optimized standard attention kernels, the absolute gain may be less dramatic than in long-sequence training. FlashAttention shines most in regimes with long sequences, high memory pressure, and HBM bandwidth bottlenecks. This matches the empirical story across the original paper and later versions. [arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

* * *

10\. 面试高频追问与高分回答
----------------

**问题一：FlashAttention 为什么显存从二次降到线性，但时间复杂度没有降到线性？**  
**高分回答：因为它避免了显式存储  $N\times N$  的中间矩阵，所以额外内存可以做到  $O(N)$ ；但 dense exact attention 仍然需要处理所有 query-key 对，所以算术 FLOPs 仍是  $O(N^2d)$ 。也就是说，它优化的是存储与 IO，而不是把所有 pairwise interaction 从数学上删掉。** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**Question 1: Why does FlashAttention reduce memory from quadratic to linear, but not time complexity to linear?**  
**High-scoring answer: because it avoids explicitly storing the  $N\times N$  intermediate matrices, so the additional memory can be reduced to  $O(N)$ ; however, dense exact attention still has to process all query-key pairs, so the arithmetic FLOPs remain  $O(N^2d)$ . In other words, it optimizes storage and IO, not the underlying pairwise interaction structure.** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**问题二：FlashAttention 的关键数学技巧是什么？**  
**高分回答：关键是 online softmax，也就是对每一行维护 running max 和 running normalization sum。这样不同 block 的 softmax 结果可以稳定且精确地合并，所以 attention 可以被重排为分块流式计算，而不引入近似误差。** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**Question 2: What is the key mathematical trick behind FlashAttention?**  
**High-scoring answer: the key is online softmax, namely maintaining a running row-wise maximum and normalization sum. That allows the softmax results from different blocks to be merged stably and exactly, so attention can be reordered into a blockwise streaming computation without approximation error.** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**问题三：为什么 FlashAttention 适合 GPU，而不是一个单纯的理论技巧？**  
**高分回答：因为 GPU 的性能高度受 memory hierarchy 影响。HBM 容量大但慢，SRAM / shared memory 小但快。FlashAttention 通过 tiling、kernel fusion 和重算，把大部分中间结果限制在片上，从而减少最昂贵的 HBM 读写，所以它是典型的 hardware-aware / IO-aware 优化。** [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**Question 3: Why is FlashAttention particularly suited to GPUs rather than being just a theoretical trick?**  
**High-scoring answer: because GPU performance is heavily shaped by the memory hierarchy. HBM is large but slow, whereas SRAM / shared memory is small but fast. FlashAttention uses tiling, kernel fusion, and recomputation to keep most intermediates on-chip, thereby reducing the most expensive HBM traffic. That makes it a classic hardware-aware and IO-aware optimization.** [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**问题四：FlashAttention-2 比 FlashAttention-1 主要强在哪？**  
**高分回答：FA1 主要解决了“别把大 attention matrix 写回 HBM”；FA2 进一步解决“如何让 GPU 并行更充分、让 Tensor Cores 更忙”。具体包括减少 non-matmul FLOPs、跨 thread blocks 并行单个 head，以及降低 shared memory 通信。** [arXiv+1](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

**Question 4: What is the main improvement of FlashAttention-2 over FlashAttention-1?**  
**High-scoring answer: FA1 mainly solves “do not write the huge attention matrix back to HBM,” while FA2 further solves “how to expose more parallelism and keep Tensor Cores busier.” Concretely, it reduces non-matmul FLOPs, parallelizes even a single head across thread blocks, and reduces shared-memory communication.** [arXiv+1](https://arxiv.org/abs/2307.08691?utm_source=chatgpt.com)

**问题五：FlashAttention-3 的关键词是什么？**  
**高分回答：Hopper、asynchrony、WGMMA、TMA、FP8。也就是利用 Hopper 新硬件特性，把搬运和计算重叠，把 matmul 与 softmax 做更深流水，并在低精度下兼顾速度和精度。** [Tri Dao+1](https://tridao.me/blog/2024/flash3/)

**Question 5: What are the keywords for FlashAttention-3?**  
**High-scoring answer: Hopper, asynchrony, WGMMA, TMA, and FP8. In other words, it exploits Hopper-specific hardware features to overlap movement and compute, pipeline matmul with softmax more aggressively, and balance speed with accuracy in low precision.** [Tri Dao+1](https://tridao.me/blog/2024/flash3/)

**问题六：FlashAttention 有什么代价？**  
**高分回答：代价主要有两个。第一，它需要更复杂的 kernel 实现与硬件适配；第二，它在 backward 中依赖重算，所以是“多一点算术，少很多内存和 IO”的 trade-off。这个 trade-off 在 GPU 上通常划算，但不是零成本。** [arXiv+1](https://arxiv.org/pdf/2205.14135)

**Question 6: What are the costs of FlashAttention?**  
**High-scoring answer: there are mainly two costs. First, it requires much more sophisticated kernels and hardware-specific tuning. Second, backward relies on recomputation, so the trade-off is “more arithmetic for much less memory and IO.” On GPUs this is usually favorable, but it is not cost-free.** [arXiv+1](https://arxiv.org/pdf/2205.14135)

* * *

11\. 一段可以直接背诵的面试满分答案
--------------------

**你可以这样答：FlashAttention 的核心不是近似 attention，而是把 exact attention 做成 IO-aware 的分块算法。标准 attention 的问题不只是  $N^2$  计算，更是会显式物化  $N\times N$  的 score 和 probability 矩阵，导致 HBM 读写和显存占用非常重。FlashAttention 通过 tiling、online softmax、kernel fusion 和 backward recomputation，把中间结果尽量留在 SRAM / shared memory 中，因此在保持输出完全等价于**

$$
\mathrm{softmax}(QK^\top)V
$$

**的前提下，把额外内存降到  $O(N)$ ，并显著减少 HBM accesses。FlashAttention-2 进一步优化并行划分和 non-matmul FLOPs，让 GPU 利用率更高；FlashAttention-3 则面向 Hopper，引入异步流水、TMA、WGMMA 和更好的 FP8 路径。所以它的本质不是改模型，而是把 attention kernel 做到更符合 GPU 硬件。** [GitHub+3arXiv+3arXiv+3](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**You can answer like this: the core of FlashAttention is not approximating attention, but turning exact attention into an IO-aware tiled algorithm. The problem with standard attention is not only the  $N^2$  computation; it is also the explicit materialization of the  $N\times N$  score and probability matrices, which causes heavy HBM traffic and memory usage. FlashAttention uses tiling, online softmax, kernel fusion, and backward recomputation to keep intermediates in SRAM / shared memory as much as possible. As a result, while remaining exactly equivalent to**

$$
\mathrm{softmax}(QK^\top)V
$$

**it reduces additional memory to  $O(N)$  and significantly cuts HBM accesses. FlashAttention-2 further improves work partitioning and reduces non-matmul FLOPs to raise GPU utilization; FlashAttention-3 targets Hopper with asynchronous pipelining, TMA, WGMMA, and a better FP8 path. So fundamentally, it does not change the model; it makes the attention kernel much more aligned with GPU hardware.** [GitHub+3arXiv+3arXiv+3](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

* * *

12\. 最后给你一个“面试官最想听到的总结”
-----------------------

**一句话压轴：FlashAttention 的第一性原理是“在不改变 exact attention 数学定义的前提下，最小化跨存储层级的数据搬运”。所以它是 attention 的系统优化，不是 attention 的近似替代。真正理解这一点，你就能把 FlashAttention、FlashAttention-2、FlashAttention-3 讲成一条很清晰的演进链：从 IO-aware，到 better parallelism，再到 hardware-specific asynchrony and low precision。** [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

**A strong closing line is this: the first-principles view of FlashAttention is to minimize data movement across the memory hierarchy without changing the mathematical definition of exact attention. So it is a systems optimization of attention, not an approximate replacement for attention. Once you understand that, FlashAttention, FlashAttention-2, and FlashAttention-3 become a clean evolution path: from IO-awareness, to better parallelism, to hardware-specific asynchrony and low precision.** [arXiv+3arXiv+3arXiv+3](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

我可以下一条继续给你整理 **“FlashAttention 面试题 30 问 30 答，严格中英对照版”**。

---

**Sources:**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)
- [GitHub - Dao-AILab/flash-attention: Fast and memory-efficient exact attention · GitHub](https://github.com/dao-ailab/flash-attention)
- [ FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision | Tri Dao ](https://tridao.me/blog/2024/flash3/)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)