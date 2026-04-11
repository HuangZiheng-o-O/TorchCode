# KV Cache 解析

**Created:** 4/10/2026 21:00:07  
**Updated:** 4/10/2026 21:01:48  
**Exported:** 4/10/2026 21:01:54  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99d16-d0fc-83ea-99e0-9b5e8b957d7b](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99d16-d0fc-83ea-99e0-9b5e8b957d7b)  

## Prompt:
4/10/2026, 9:00:07 PM

        
KV cache 是什么？为什么能显著提升推理速度？代价是什么？  think harder

## Response:
4/10/2026, 9:00:09 PM

> Thought for a couple of seconds

KV cache 本质上是：在 Transformer 自回归推理时，把历史 token 已经计算过的 **Key** 和 **Value** 向量缓存下来，后续生成新 token 时不再重复计算这些历史位置的注意力投影，因此把“每生成一个 token 都重新跑一遍整段前缀”的代价，变成“只对新 token 做一次增量计算，再与历史缓存做注意力交互”。它显著提升推理速度的根本原因，不是减少了注意力本身对历史序列的访问，而是**消除了对历史 token 在每一层反复做线性投影和前向传播的重复计算**。代价则主要是**显存/内存占用显著增加、带宽压力变大、batch 扩展受限，以及长上下文下缓存管理更复杂**。  
The essence of a KV cache is this: during autoregressive Transformer inference, we store the **Key** and **Value** vectors of previous tokens after they are computed, so when generating the next token we do not recompute attention projections for the entire history. As a result, the cost changes from “re-running the whole prefix for every new token” to “doing incremental computation only for the new token, then attending to the cached history.” The main reason it speeds up inference is not that attention no longer looks at past tokens, but that it **eliminates repeated forward computation and repeated linear projections for historical tokens across all layers**. The trade-offs are mainly **much higher memory usage, greater memory-bandwidth pressure, reduced batch scalability, and more complicated cache management for long contexts**.

先给出标准数学形式。对第  $l$  层，输入隐状态记为  $H^{(l)} \in \mathbb{R}^{T \times d}$ ，则注意力中的 Query、Key、Value 为：  
Let me first give the standard mathematical form. For layer  $l$ , let the input hidden states be  $H^{(l)} \in \mathbb{R}^{T \times d}$ . Then the Query, Key, and Value in attention are:

$$
Q^{(l)} = H^{(l)} W_Q^{(l)}, \quad K^{(l)} = H^{(l)} W_K^{(l)}, \quad V^{(l)} = H^{(l)} W_V^{(l)}
$$

中文解释：这里  $T$  是当前序列长度， $d$  是隐藏维度， $W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}$  是第  $l$  层的投影矩阵。它们的作用是把每个 token 的表示映射到注意力机制所需的 Query、Key、Value 空间。  
English explanation: Here,  $T$  is the current sequence length,  $d$  is the hidden dimension, and  $W_Q^{(l)}, W_K^{(l)}, W_V^{(l)}$  are the projection matrices of layer  $l$ . Their role is to map each token representation into the Query, Key, and Value spaces required by the attention mechanism.

对应的因果自注意力输出可以写成：  
The corresponding causal self-attention output can be written as:

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

中文解释： $d_k$  是每个头的 Key 维度， $M$  是 causal mask，用于禁止当前位置看到未来 token。这个公式的作用是：当前 token 的 Query 与所有可见历史位置的 Key 做匹配，得到权重，再对对应的 Value 做加权求和。  
English explanation:  $d_k$  is the per-head key dimension, and  $M$  is the causal mask that prevents a position from attending to future tokens. The role of this formula is that the current token’s Query is matched against all visible historical Keys to produce attention weights, which are then used to compute a weighted sum of the corresponding Values.

不使用 KV cache 时，假设我们已经有前缀长度  $t-1$ ，现在要生成第  $t$  个 token。虽然我们真正只新增了一个 token，但实际通常会把长度为  $t$  的整段前缀再次送入模型，于是每一层都会重新计算前  $t$  个位置的  $Q,K,V$  以及后续前向过程。也就是说，历史 token 明明没有变，却被重复编码很多次。  
Without KV cache, suppose we already have a prefix of length  $t-1$ , and now we want to generate the  $t$ \-th token. Even though only one new token is added, the entire length- $t$  prefix is typically fed through the model again, so every layer recomputes  $Q, K, V$  for all  $t$  positions, along with the downstream forward computation. In other words, historical tokens do not change, yet they are redundantly re-encoded many times.

使用 KV cache 后，在第  $t$  步，我们只需要对**最新 token**计算当前层的  $q_t, k_t, v_t$ ，然后把  $k_t, v_t$  追加到缓存中：  
With KV cache, at step  $t$ , we only need to compute the current layer’s  $q_t, k_t, v_t$  for the **newest token**, and then append  $k_t, v_t$  to the cache:

$$
K^{(l)}_{\text{cache}, t} = \big[K^{(l)}_{\text{cache}, t-1};\; k_t^{(l)}\big], \quad V^{(l)}_{\text{cache}, t} = \big[V^{(l)}_{\text{cache}, t-1};\; v_t^{(l)}\big]
$$

中文解释： $K^{(l)}_{\text{cache}, t-1}$  和  $V^{(l)}_{\text{cache}, t-1}$  分别表示第  $l$  层在前  $t-1$  个位置上已经缓存的 Key 和 Value； $[\,;\,]$  表示在时间维上拼接。它们的作用是把历史上下文的注意力状态持久保留下来，供后续 token 直接复用。  
English explanation:  $K^{(l)}_{\text{cache}, t-1}$  and  $V^{(l)}_{\text{cache}, t-1}$  denote the cached Keys and Values for the first  $t-1$  positions at layer  $l$ ;  $[\,;\,]$  means concatenation along the time dimension. Their role is to preserve the attention state of the historical context so future tokens can reuse it directly.

然后当前 token 的注意力只需要对缓存做一次查询：  
Then the current token only needs to query the cache once:

$$
o_t^{(l)} = \mathrm{softmax}\!\left(\frac{q_t^{(l)} \left(K^{(l)}_{\text{cache}, t}\right)^\top}{\sqrt{d_k}}\right) V^{(l)}_{\text{cache}, t}
$$

中文解释：这里  $q_t^{(l)}$  是当前 token 在第  $l$  层的 Query， $o_t^{(l)}$  是该层注意力输出。这个公式表明：第  $t$  步并不需要重新构造历史所有 token 的  $Q,K,V$ ，只需用当前 Query 去读取历史缓存。  
English explanation: Here,  $q_t^{(l)}$  is the Query of the current token at layer  $l$ , and  $o_t^{(l)}$  is the attention output of that layer. This formula shows that at step  $t$ , we do not need to reconstruct the historical tokens’  $Q,K,V$ ; we only use the current Query to read from the historical cache.

这就是为什么 KV cache 能显著提速。面试里最容易答浅的地方是只说“避免重复计算”，但更高分的说法是：**它避免的不是单纯的 softmax，而是整条历史路径在每一层上的重复前向**。对于一个  $L$  层模型，若不缓存，第  $t$  步会对长度  $t$  的全部前缀做  $L$  层计算；而缓存后，第  $t$  步只对新 token 做  $L$  层增量计算，再读历史缓存。因此，从“每步都重算前缀”的角度看，计算量从总生成长度上的近似二次累积，下降为更接近“每步一次增量前向 + 一次对历史缓存的注意力读取”。  
This is exactly why KV cache gives such a large speedup. A shallow interview answer would merely say “it avoids redundant computation,” but a stronger answer is: **what it avoids is not just the softmax itself, but repeated full-prefix forward propagation across every layer**. For an  $L$ \-layer model, without caching, step  $t$  runs all  $L$  layers over the entire prefix of length  $t$ ; with caching, step  $t$  only performs incremental computation for the new token through those  $L$  layers, and then reads from the historical cache. So from the perspective of generation over many steps, the cost drops from a near-quadratic accumulation over generated length to something much closer to “one incremental forward pass per step plus one attention read over cached history.”

更严谨一点地说，很多人会把复杂度简单记成“从  $O(T^2)$  变成  $O(T)$ ”，这个说法方向对，但不够精确。更准确地说：**对生成整个长度为  $T$  的序列而言**，不使用 cache 时，每一步都要重新处理已有前缀，总计算会累积成更高阶的重复代价；使用 cache 后，每一步只新增一个位置的投影和 MLP 计算，因此大部分重复项被消除。不过，当前 token 依然要和所有历史 Key 做点积，所以**注意力读取历史的那部分并没有消失**，它仍然随上下文长度增长。  
To be more precise, people often summarize the complexity change as “from  $O(T^2)$  to  $O(T)$ ,” which is directionally correct but not exact enough. A better statement is this: **for generating an entire sequence of length  $T$ **, without cache, every step reprocesses the existing prefix, so total computation accumulates large repeated costs; with cache, each step only adds one new position’s projections and MLP computation, so most repeated terms are removed. However, the current token still needs to take dot products with all historical Keys, so **the part of attention that reads from history does not disappear**; it still grows with context length.

为什么这在实践中会“显著”提速，而不是“略微”提速？因为大模型推理通常是逐 token 解码，尤其在 batch 不大、decode 阶段占主导时，重复处理前缀是最大的浪费之一。KV cache 把这部分最明显的重复去掉了，所以吞吐和延迟都会改善。尤其是上下文越长，收益越大，因为历史越长，重复计算越夸张。  
Why is the speedup “significant” in practice rather than merely “modest”? Because large-model inference is usually done token by token during decoding, and especially when batch size is small and the decode phase dominates, repeatedly processing the prefix is one of the biggest sources of waste. KV cache removes this most obvious redundancy, so both throughput and latency improve. The longer the context, the larger the gain, because the longer the history, the worse the redundant computation would otherwise be.

但它的代价也非常明确，第一是显存/内存占用。因为你要为**每一层、每个 token、每个头**保存 Key 和 Value。一个常见的缓存内存估算公式是：  
But its trade-offs are also very clear. The first is memory usage, because you must store Keys and Values for **every layer, every token, and every head**. A common memory estimate for the cache is:

$$
\text{KV Cache Memory} \approx 2 \times L \times T \times H \times d_{\text{head}} \times \text{bytes\_per\_element}
$$

中文解释：其中  $2$  表示同时存 Key 和 Value， $L$  是层数， $T$  是缓存的序列长度， $H$  是注意力头数， $d_{\text{head}}$  是每个头的维度， $\text{bytes\_per\_element}$  由数据类型决定，比如 FP16 通常是 2 字节。这个公式的作用是说明：KV cache 的内存开销会随着上下文长度线性增长，而且系数非常大。  
English explanation: Here,  $2$  means we store both Key and Value,  $L$  is the number of layers,  $T$  is the cached sequence length,  $H$  is the number of attention heads,  $d_{\text{head}}$  is the dimension of each head, and  $\text{bytes\_per\_element}$  depends on the datatype, for example FP16 is usually 2 bytes. The role of this formula is to show that KV cache memory grows linearly with context length, and the constant factor is quite large.

第二个代价是**内存带宽瓶颈**。很多人以为 cache 以后计算少了就一定更快，但在现代 GPU 上，decode 阶段常常不是纯算力受限，而是**读缓存受限**。因为每生成一个 token，都要从很多层里把长历史的  $K,V$  读出来参与计算。也就是说，KV cache 用“减少 FLOPs”换来了“更多持续内存读取”。当上下文极长时，这种带宽压力会非常明显。  
The second trade-off is **memory-bandwidth bottleneck**. Many people assume that once computation is reduced, inference must become faster, but on modern GPUs the decode phase is often not purely compute-bound; it is frequently **limited by reading the cache**. For every new token, the model needs to read long-history  $K,V$  tensors from many layers. In other words, KV cache trades “fewer FLOPs” for “more sustained memory reads.” When the context becomes very long, this bandwidth pressure becomes substantial.

第三个代价是**batch 能力下降**。显存被 KV cache 占掉后，可容纳的并发请求数会下降，所以系统级吞吐不一定单调变好。单条请求的延迟往往变低，但如果服务端非常关注整体 QPS，cache 带来的显存膨胀可能压缩 batch size，进而影响总体吞吐。  
The third trade-off is **reduced batch capacity**. Once GPU memory is consumed by KV cache, the number of concurrent requests that can fit decreases, so system-level throughput does not always improve monotonically. The latency of a single request often becomes lower, but if the serving system cares heavily about overall QPS, the memory inflation caused by caching can shrink the achievable batch size and thus affect total throughput.

第四个代价是**长上下文下的工程复杂度**。例如分页式 KV cache、连续/非连续内存管理、prefix sharing、cache eviction、multi-turn session 复用、以及多卡场景下的 cache 分片，这些都不是理论问题，而是实际部署中的核心难点。面试里说到这里，通常会显得你不只懂算法，还懂推理系统。  
The fourth trade-off is **engineering complexity under long contexts**. Examples include paged KV cache, contiguous versus non-contiguous memory management, prefix sharing, cache eviction, multi-turn session reuse, and cache sharding across multiple devices. These are not just theoretical issues; they are core deployment challenges in real inference systems. Mentioning this in an interview usually shows that you understand not only the algorithm, but also inference systems.

还有一个容易被忽略的点：KV cache 只在 **inference，尤其是 autoregressive decoding** 中最有效，在训练阶段通常不能直接这么用。因为训练时要并行处理整段序列、要做反向传播，计算图和内存需求完全不同。  
There is also an easily overlooked point: KV cache is most effective only in **inference, especially autoregressive decoding**, and it is generally not used in the same way during training. During training, entire sequences are processed in parallel and backpropagation is required, so the computation graph and memory requirements are fundamentally different.

如果要给出一句非常面试化、非常高分的总结，我会这样说：KV cache 的核心价值，是把 Transformer 解码从“反复重算历史”改成“历史一次计算、后续反复读取”，因此显著降低了自回归生成的重复计算；但它并没有消除对长历史的注意力访问，所以系统性能最终会越来越受显存容量和内存带宽约束。  
If I had to give a concise, highly interview-worthy summary, I would say this: the core value of KV cache is that it turns Transformer decoding from “recomputing history again and again” into “compute history once, then repeatedly read it,” which dramatically reduces redundant work in autoregressive generation; however, it does not eliminate attention access to long histories, so system performance eventually becomes increasingly constrained by memory capacity and memory bandwidth.

最后补一句区分度很高的话：**prefill 阶段和 decode 阶段对 KV cache 的收益不同**。prefill 是把已有 prompt 一次性编码进去，这一阶段主要还是大矩阵计算；而 decode 是一次只生成一个 token，这时 KV cache 的价值最大，因为它最有效地避免了历史前缀被重复前向。  
Let me add one final high-signal point: **the benefit of KV cache differs between the prefill phase and the decode phase**. Prefill processes the existing prompt in bulk, so it is still dominated by large matrix computation; decode generates one token at a time, and that is where KV cache is most valuable, because it most effectively prevents repeated forward passes over the historical prefix.

面试里被追问时，你可以用一句话收尾：**KV cache 加速的是“增量生成”，代价是“用空间和带宽换时间”。**  
If you are pressed in an interview, you can close with one sentence: **KV cache accelerates “incremental generation,” and its trade-off is “using memory space and bandwidth to buy time.”**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)