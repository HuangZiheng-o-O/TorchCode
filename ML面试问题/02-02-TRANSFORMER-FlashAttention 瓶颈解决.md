# FlashAttention 瓶颈解决

**Created:** 4/10/2026 21:00:17  
**Updated:** 4/10/2026 21:01:47  
**Exported:** 4/10/2026 21:01:59  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99d1f-f7e0-83ea-b480-ea6952f09d8d](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99d1f-f7e0-83ea-b480-ea6952f09d8d)  

## Prompt:
4/10/2026, 9:00:16 PM

        
FlashAttention 解决了什么瓶颈？  think harder

## Response:
4/10/2026, 9:00:18 PM

我先直接给你面试级答案，再补上瓶颈、机制和为什么它有效。  
I’ll give you the interview-grade answer first, then unpack the bottleneck, the mechanism, and why it works.

> Thought for 6s

FlashAttention 解决的**核心瓶颈不是“注意力公式本身算得不够快”**，而是**标准 attention 在 GPU 上严重受限于显存读写（memory IO / bandwidth），尤其是反复把巨大的 attention matrix  $S=QK^\top$  和 softmax 后的概率矩阵  $P$  写回 HBM 再读回来**。它的本质贡献是：把 attention 改写成 **IO-aware 的分块计算**，尽量在片上 SRAM / shared memory 里完成中间计算，从而把标准 attention 的**二次显存开销降到线性级别**，同时获得实际 wall-clock 加速，而且它是 **exact attention，不是近似 attention**。[arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

The **core bottleneck FlashAttention solves is not that the attention formula itself lacks arithmetic speed**. The real issue is that **standard attention is heavily limited by GPU memory IO / bandwidth**, especially because it repeatedly writes the large attention matrix  $S=QK^\top$  and the post-softmax probability matrix  $P$  to HBM and then reads them back. Its key contribution is to reformulate attention as an **IO-aware tiled algorithm** that keeps intermediate computation on-chip in SRAM / shared memory as much as possible. This reduces the **quadratic memory footprint to linear**, while also delivering real wall-clock speedup, and it remains **exact attention rather than approximate attention**. [arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

标准 self-attention 的计算写成：  
Standard self-attention can be written as:

$$
S = QK^\top,\qquad P=\mathrm{softmax}(S),\qquad O = PV
$$

中文解释：这里  $Q,K,V \in \mathbb{R}^{N \times d}$ ， $N$  是序列长度， $d$  是 head dimension。 $S\in\mathbb{R}^{N\times N}$  是 attention score matrix， $P\in\mathbb{R}^{N\times N}$  是 softmax 后的注意力权重， $O\in\mathbb{R}^{N\times d}$  是输出。这个公式说明了标准 attention 会显式生成  $N\times N$  级别的中间结果。  
English explanation: Here  $Q,K,V \in \mathbb{R}^{N \times d}$ , where  $N$  is the sequence length and  $d$  is the head dimension.  $S\in\mathbb{R}^{N\times N}$  is the attention score matrix,  $P\in\mathbb{R}^{N\times N}$  is the post-softmax attention weight matrix, and  $O\in\mathbb{R}^{N\times d}$  is the output. This formula shows why standard attention explicitly materializes  $N\times N$ \-scale intermediates.

真正的问题在于：**理论上 attention 的 FLOPs 很多，但在现代 GPU 上，很多时候先撞上的不是算力上限，而是显存带宽上限**。标准实现通常先算出  $QK^\top$ ，把整个  $S$  存到 HBM；再读出来做 softmax；再把  $P$  写回 HBM；最后再读  $P$  去乘  $V$ 。所以瓶颈不只是 “ $O(N^2)$ ”，而是 **对大矩阵的反复 materialization 和 HBM 往返读写**。FlashAttention 论文明确指出，attention 需要做 **IO-aware** 设计，也就是优化 GPU 不同层级内存之间的数据搬运，而不只是盯着算术复杂度。[arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

The real issue is that **although attention has many FLOPs, on modern GPUs the first hard limit is often memory bandwidth rather than raw compute throughput**. A standard implementation usually computes  $QK^\top$ , stores the full  $S$  in HBM, reads it back for softmax, writes  $P$  back to HBM, and then reads  $P$  again to multiply by  $V$ . So the bottleneck is not merely “ $O(N^2)$ ” in an abstract sense, but **repeated materialization of large matrices and repeated HBM traffic**. The FlashAttention paper explicitly argues that attention must be designed in an **IO-aware** way, meaning the movement of data across the GPU memory hierarchy must be optimized, not just the arithmetic complexity. [arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

你在面试里可以直接说一个非常加分的判断：**FlashAttention 优化的是“访存复杂度”而不是“数学定义”**。它没有把 attention 改成线性 attention，也没有牺牲精度做近似；它还是在算同一个 softmax attention，只是把计算顺序重新组织成 block-wise / tiled 的形式，让  $Q$ 、 $K$ 、 $V$  分块加载到 SRAM 中，在块内完成 score、softmax 和与  $V$  的乘法，再把结果累积到输出，而不是把整个  $N\times N$  的中间矩阵落到 HBM。[arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

A strong interview-level statement is: **FlashAttention optimizes memory-access complexity rather than changing the mathematical definition of attention**. It does not replace attention with linear attention, and it does not trade exactness for approximation. It still computes the same softmax attention, but it reorganizes the execution into a block-wise / tiled form:  $Q$ ,  $K$ , and  $V$  are loaded into SRAM in tiles, scores, softmax, and the multiplication with  $V$  are computed within the tile, and the result is accumulated directly into the output, instead of materializing the full  $N\times N$  intermediate matrices in HBM. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

它之所以能做到这一点，关键在于 **online softmax**。对于某一行 attention，如果分块处理 keys，不需要先看到整行所有分数再一次性做 softmax；可以维护运行中的最大值和归一化因子，边扫描边更新。典型地可写成：  
It achieves this through **online softmax**. For a given attention row, if keys are processed block by block, we do not need to see the full row of scores before computing softmax. We can maintain a running maximum and normalization factor and update them incrementally while scanning blocks. A typical form is:

$$
m^{(t)} = \max\!\big(m^{(t-1)}, \max(S^{(t)})\big)
$$
 
$$
\ell^{(t)} = e^{m^{(t-1)}-m^{(t)}}\ell^{(t-1)} + \sum_j e^{S^{(t)}_j - m^{(t)}}
$$
 
$$
O^{(t)} = \frac{ e^{m^{(t-1)}-m^{(t)}} \ell^{(t-1)} O^{(t-1)} + \sum_j e^{S^{(t)}_j-m^{(t)}} V_j }{\ell^{(t)}}
$$

中文解释：这里  $t$  表示处理到第  $t$  个 key/value block， $m^{(t)}$  是当前扫描到的全局最大 score，用来保证 softmax 数值稳定； $\ell^{(t)}$  是当前累计的 softmax 分母； $O^{(t)}$  是当前累计输出。这个递推的作用是：**不用存整行  $S$  或  $P$ ，也能精确得到最终 softmax attention 的结果**。  
English explanation: Here  $t$  denotes the  $t$ \-th key/value block,  $m^{(t)}$  is the running global maximum score used for numerical stability,  $\ell^{(t)}$  is the running softmax denominator, and  $O^{(t)}$  is the running output. The role of this recurrence is that **we can obtain the exact final softmax-attention result without storing the full row of  $S$  or  $P$ **.

所以它解决的第一个大瓶颈是**显存容量瓶颈**。标准 attention 通常需要显式存  $N\times N$  的中间结果，因此 memory footprint 随序列长度二次增长；FlashAttention 通过不 materialize 整个 attention matrix，把额外内存从二次级别降到了近似线性级别，这也是长上下文训练和推理能落地的关键原因之一。原论文把它概括为比标准 attention 更少的 HBM accesses，并且在一定 SRAM 大小范围内是 IO-optimal 的。[arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

So the first major bottleneck it solves is the **memory-capacity bottleneck**. Standard attention usually needs to explicitly store  $N\times N$  intermediates, so the memory footprint grows quadratically with sequence length. FlashAttention avoids materializing the full attention matrix, reducing the extra memory from quadratic scale to approximately linear scale. This is one of the key reasons long-context training and inference become much more practical. The original paper summarizes this as using fewer HBM accesses than standard attention and being IO-optimal over a range of SRAM sizes. [arXiv+1](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

它解决的第二个大瓶颈是**真实运行时间瓶颈**。很多早期“高效 attention”方法虽然在理论复杂度上做了文章，但实际 wall-clock speedup 不明显，因为 GPU 上真正慢的是数据搬运。FlashAttention 的价值恰恰在于它抓住了最贵的环节——HBM 读写——所以能带来真实端到端加速，而不是只在大  $O$  记号上更漂亮。论文报告它在多个训练场景上带来了显著加速，FlashAttention-2 进一步指出，原始 FlashAttention 仍未充分利用 GPU，后续通过更好的 work partitioning、降低非 matmul FLOPs、提升 occupancy，又把效率继续推高。[arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

The second major bottleneck it solves is the **real runtime bottleneck**. Many earlier “efficient attention” methods improved theoretical complexity, yet delivered limited wall-clock gains because the real slowdown on GPUs comes from data movement. FlashAttention matters precisely because it attacks the most expensive part—HBM traffic—so it yields real end-to-end acceleration rather than just a nicer big- $O$  story. The paper reports substantial speedups across multiple training settings, and FlashAttention-2 further explains that the original FlashAttention still did not fully utilize the GPU; by improving work partitioning, reducing non-matmul FLOPs, and increasing occupancy, it pushes efficiency further. [arXiv+2arXiv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

这里有一个非常重要的“深层理解点”：**FlashAttention 没有消灭 attention 的二次计算本身，它主要消灭的是二次中间存储和大量低效访存**。因此如果面试官追问“它是不是把 attention 从  $O(N^2)$  变成  $O(N)$  了”，高分回答应该是：**不是在 FLOPs 意义上把精确 softmax attention 变成线性；它保留了 exact attention 的计算语义，但把 memory complexity 从需要存  $N^2$  中间量降到了线性级别，并显著减少了 IO，所以实际更快、更省显存。**[arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

There is a very important deeper insight here: **FlashAttention does not eliminate the quadratic compute structure of attention itself; it mainly eliminates quadratic intermediate storage and a large amount of inefficient memory traffic**. So if an interviewer asks, “Did it turn attention from  $O(N^2)$  into  $O(N)$ ?”, the high-quality answer is: **not in the FLOPs sense for exact softmax attention. It preserves the semantics of exact attention, but reduces memory complexity from storing  $N^2$  intermediates down to linear scale, while drastically reducing IO, which makes it faster and more memory-efficient in practice.** [arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

再往深一点说，FlashAttention 体现的是一个系统层面的原则：**在现代加速器上，算法性能不仅由算术复杂度决定，还由 memory hierarchy 决定**。同一个数学公式，如果执行顺序不同、kernel fusion 不同、是否显式写出中间矩阵不同，性能会差很多。FlashAttention 把 attention 从“数学上正确”推进到了“硬件上高效”，这也是它影响力很大的根本原因。[arXiv+2ar5iv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

At a deeper level, FlashAttention reflects a systems principle: **on modern accelerators, algorithmic performance is determined not only by arithmetic complexity but also by the memory hierarchy**. The same mathematical formula can have very different performance depending on execution order, kernel fusion, and whether intermediate matrices are explicitly materialized. FlashAttention takes attention from being merely “mathematically correct” to being “hardware-efficient,” and that is a major reason for its impact. [arXiv+2ar5iv+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

面试里最后可以这样收束：\*\*FlashAttention 解决的是标准 attention 的 IO 和显存瓶颈。它通过分块计算、kernel fusion 和 online softmax，避免显式存储完整 attention matrix，减少 HBM 访问，把额外显存从二次降到线性，并在不近似的前提下获得真实加速。\*\*这句话基本就是接近满分的总结。[arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

A strong final interview summary is: **FlashAttention solves the IO and memory bottlenecks of standard attention. By using tiled computation, kernel fusion, and online softmax, it avoids explicitly storing the full attention matrix, reduces HBM traffic, lowers the extra memory from quadratic to linear scale, and delivers real speedups without approximation.** That is very close to a full-score summary. [arXiv+2OpenReview+2](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)

如果你愿意，我下一条可以继续把这个问题展开成“为什么 FlashAttention 比 standard attention 更适合 GPU”的**面试口语版 90 秒回答**。

---

**Sources:**

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)