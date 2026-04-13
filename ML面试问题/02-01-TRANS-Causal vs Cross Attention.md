# Causal vs Cross Attention

**Created:** 4/10/2026 21:05:51  
**Updated:** 4/10/2026 21:06:59  
**Exported:** 4/10/2026 21:08:48  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99e6e-ae7c-83ea-aba8-223499408b79](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99e6e-ae7c-83ea-aba8-223499408b79)  

## Prompt:
4/10/2026, 9:05:51 PM

        
causal attention 和 cross attention 有什么区别？ think harder

## Response:
4/10/2026, 9:05:53 PM

> Thought for a couple of seconds

**结论先说：causal attention 和 cross attention 的本质区别，不在于“算 attention 的公式变了”，而在于“谁可以看谁”。Causal attention 是一种带因果遮罩的 self-attention，用来限制当前位置只能看当前及之前的 token；cross attention 是 query 来自一个序列、key/value 来自另一个序列，用来让一个序列有条件地读取另一个序列的信息。前者解决“不能偷看未来”，后者解决“如何利用外部上下文”。**

**The core difference is not that the attention formula changes, but that the connectivity pattern changes. Causal attention is masked self-attention, where each position can only attend to itself and previous tokens; cross attention uses queries from one sequence and keys/values from another sequence, allowing one sequence to condition on external context. Causal attention solves the “no peeking into the future” constraint, while cross attention solves “how to read information from another source.”**

从数学上看，二者都基于同一个 attention 核心公式：

From a mathematical perspective, both are built on the same core attention formula:

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

中文解释：这里  $Q$  是 query，表示“我要找什么”； $K$  是 key，表示“我这里有什么”； $V$  是 value，表示“真正被取出来并加权聚合的信息”； $d_k$  是 key 的维度，用于缩放，防止点积过大导致 softmax 过于尖锐。这个公式本身不区分 causal 还是 cross，区别来自  $Q,K,V$  的来源，以及是否加 mask。

English explanation: Here,  $Q$  is the query, representing “what I am looking for”;  $K$  is the key, representing “what is available here”;  $V$  is the value, which is the actual information being retrieved and aggregated;  $d_k$  is the key dimension used for scaling, preventing overly large dot products that make softmax too sharp. The formula itself does not distinguish causal from cross attention; the difference comes from where  $Q,K,V$  come from and whether a mask is applied.

* * *

**先讲 causal attention：它其实就是 self-attention + 下三角 mask。**

**Let’s start with causal attention: it is essentially self-attention plus a lower-triangular mask.**

在 causal attention 里， $Q$ 、 $K$ 、 $V$  都来自同一个序列，所以它首先是 self-attention；但为了保证自回归生成时第  $t$  个 token 不能看到未来的第  $t+1,t+2,\dots$  个 token，需要在 attention score 上加一个因果遮罩：

In causal attention,  $Q$ ,  $K$ , and  $V$  all come from the same sequence, so it is first a form of self-attention. But to ensure that, during autoregressive generation, token  $t$  cannot see future tokens  $t+1,t+2,\dots$ , we add a causal mask to the attention scores:

$$
\mathrm{CausalAttention}(X)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

其中  $Q=XW_Q,\;K=XW_K,\;V=XW_V$ 。mask  $M$  的定义通常是：若位置  $j>i$ ，则  $M_{ij}=-\infty$ ；否则  $M_{ij}=0$ 。

Here,  $Q=XW_Q,\;K=XW_K,\;V=XW_V$ . The mask  $M$  is typically defined as: if position  $j>i$ , then  $M_{ij}=-\infty$ ; otherwise  $M_{ij}=0$ .

中文解释： $X$  是输入序列表示， $W_Q,W_K,W_V$  是可学习参数矩阵。这个 mask 的作用是把未来位置的 attention score 压成 0，因此当前位置只能聚合历史信息。它的意义非常关键：训练时强制模型遵守时间因果性，推理时才能逐 token 生成而不发生信息泄漏。

English explanation:  $X$  is the input sequence representation, and  $W_Q,W_K,W_V$  are learnable parameter matrices. The role of the mask is to force attention scores for future positions to zero, so the current position can only aggregate historical information. This is crucial because it enforces temporal causality during training, making token-by-token generation valid at inference time without information leakage.

你可以把它理解成：**causal attention 的重点不是“跨序列交互”，而是“单序列内部的时间方向约束”。** 它最典型地出现在 GPT 这类 decoder-only 模型里。

You can think of it this way: **the emphasis of causal attention is not cross-sequence interaction, but directional constraints within a single sequence.** It is the standard mechanism in decoder-only models such as GPT.

* * *

**再讲 cross attention：它的核心不是 mask，而是 Q 和 K/V 来自不同序列。**

**Now for cross attention: its defining feature is not the mask, but that  $Q$  and  $K/V$  come from different sequences.**

在 cross attention 中，query 来自目标序列，key 和 value 来自源序列：

In cross attention, the query comes from the target sequence, while the key and value come from the source sequence:

$$
Q=X_{\text{target}}W_Q,\quad K=X_{\text{source}}W_K,\quad V=X_{\text{source}}W_V
$$

进一步写成 attention 形式就是：

Written in attention form, it becomes:

$$
\mathrm{CrossAttention}(X_{\text{target}},X_{\text{source}}) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

中文解释： $X_{\text{target}}$  是“当前正在更新的序列表示”，比如 decoder 当前层的 hidden states； $X_{\text{source}}$  是“外部条件信息”，比如 encoder 输出、图像特征、检索到的文档表示等。cross attention 的作用不是建模目标序列内部关系，而是让目标序列按需读取源序列信息。

English explanation:  $X_{\text{target}}$  is the sequence currently being updated, such as decoder hidden states at the current layer;  $X_{\text{source}}$  is the external conditioning information, such as encoder outputs, image features, or retrieved document representations. The role of cross attention is not to model internal relationships within the target sequence, but to let the target sequence selectively read from the source sequence.

这也是为什么 cross attention 常见于 encoder-decoder 架构，比如机器翻译中 decoder 生成第  $t$  个词时，会通过 cross attention 去“对齐”源语言句子中最相关的部分。多模态模型里，文本 token 也可以通过 cross attention 去读取图像 patch 的表示。

This is why cross attention is common in encoder-decoder architectures. For example, in machine translation, when the decoder generates the  $t$ \-th word, it uses cross attention to align with the most relevant parts of the source sentence. In multimodal models, text tokens can also use cross attention to read image patch representations.

* * *

**所以一句话区分：causal attention 关注“访问权限沿时间轴受限”，cross attention 关注“信息来源来自另一条序列”。**

**So in one sentence: causal attention is about restricted access along the time axis, while cross attention is about reading information from another sequence.**

更进一步地说，二者是**两个正交维度**，不是互斥概念。也就是说，一个 attention 层可以是 self-attention 或 cross attention；同时它还可以是 masked 或 unmasked。

Going one step further, these are **two orthogonal dimensions**, not mutually exclusive concepts. An attention layer can be self-attention or cross attention, and it can also be masked or unmasked.

例如，decoder 里通常会有两种 attention：先做 **causal self-attention**，保证当前 token 只能看过去；再做 **cross attention**，去读取 encoder 输出。这两层往往是连续出现的。

For example, a decoder typically contains two types of attention: first **causal self-attention**, ensuring the current token only sees the past; then **cross attention**, which reads from encoder outputs. These two layers often appear sequentially.

* * *

**面试里最容易加分的一点，是说明它们解决的问题完全不同。**

**A strong interview point is to explain that they solve fundamentally different problems.**

causal attention 解决的是训练目标和推理过程的一致性问题。语言模型训练时虽然能并行计算所有位置，但每个位置只能依赖左侧上下文，这样训练分布才和生成分布一致。否则模型训练时偷看未来，推理时却看不到未来，就会产生 exposure mismatch。

Causal attention solves the consistency problem between the training objective and the inference procedure. Although language models can compute all positions in parallel during training, each position must depend only on left context so that the training distribution matches the generation distribution. Otherwise, the model would peek at future tokens during training but not at inference, causing an exposure mismatch.

cross attention 解决的是条件建模问题。目标序列不是只靠自己生成，而是在“给定外部输入”的条件下生成。没有 cross attention，decoder 很难动态选择源序列中哪些部分最相关；有了 cross attention，它就能做软对齐和按需检索。

Cross attention solves the conditional modeling problem. The target sequence is not generated solely from itself, but conditioned on external input. Without cross attention, the decoder has difficulty dynamically selecting which parts of the source sequence are relevant; with cross attention, it can perform soft alignment and selective retrieval.

* * *

**从信息流角度看，causal attention 是“序列内部流动，但单向”；cross attention 是“序列之间流动”。**

**From the perspective of information flow, causal attention is “within-sequence but one-way,” while cross attention is “across sequences.”**

如果把一个序列看成一张图，causal attention 对应的是下三角邻接关系，即节点  $i$  只能连接到  $j \le i$  的节点；而 cross attention 对应的是二部图连接，即目标序列中的节点去连接源序列中的节点。这个视角非常直观，也很适合面试解释。

If you view a sequence as a graph, causal attention corresponds to a lower-triangular adjacency structure, where node  $i$  can only connect to nodes  $j \le i$ ; cross attention corresponds to a bipartite connection pattern, where nodes in the target sequence connect to nodes in the source sequence. This viewpoint is intuitive and works very well in interviews.

* * *

**再补一个常见误区：很多人会把 cross attention 理解成“两个序列互相看”，这个说法不够准确。**

**One common misconception is to describe cross attention as “two sequences attending to each other,” but that is not precise.**

严格来说，标准 cross attention 是**单向的**：目标序列用自己的 query 去读取源序列的 key/value。它不意味着源序列也同时读取目标序列。若要双向交互，通常要么堆叠两次反向的 cross attention，要么设计更复杂的 co-attention / bidirectional interaction 模块。

Strictly speaking, standard cross attention is **one-way**: the target sequence uses its own queries to read the source sequence’s keys/values. It does not imply that the source sequence simultaneously reads from the target sequence. For bidirectional interaction, one usually stacks another reverse cross attention layer or designs a more complex co-attention / bidirectional interaction module.

* * *

**如果面试官问“能不能用一个例子快速区分”，最好的回答是：GPT 用 causal attention，机器翻译的 Transformer decoder 既用 causal self-attention，也用 cross attention。**

**If the interviewer asks for a quick example, the best answer is: GPT uses causal attention, while the Transformer decoder in machine translation uses both causal self-attention and cross attention.**

GPT 生成下一个词时，只能看前面的词，所以必须 causal；而在翻译任务中，decoder 一边遵守自回归生成约束，一边还要读取 encoder 编码后的源句子表示，所以既需要 causal self-attention，也需要 cross attention。

When GPT generates the next token, it can only look at previous tokens, so causal masking is required. In translation, the decoder must both obey the autoregressive generation constraint and read the source sentence representations produced by the encoder, so it needs both causal self-attention and cross attention.

* * *

**最后总结成面试里的高分表达：causal attention 和 cross attention 使用的是同一套 attention 计算框架，但前者通过 mask 约束同一序列内部的可见范围，核心目的是保证因果性；后者通过让 query 与 key/value 来自不同序列，实现条件信息注入，核心目的是跨序列对齐与信息检索。一个强调“不能看未来”，一个强调“去看别的地方”。**

**To summarize in an interview-ready way: causal attention and cross attention use the same attention computation framework, but causal attention constrains visibility within the same sequence through masking, with the main goal of preserving causality; cross attention lets queries and keys/values come from different sequences, enabling conditional information injection, with the main goal of cross-sequence alignment and information retrieval. One emphasizes “do not look ahead,” while the other emphasizes “look somewhere else.”**

你要是想要，我下一步可以直接给你一版 **“90 秒面试口语答案”**，或者给你补上 **self-attention / masked self-attention / causal attention / cross attention 四者关系图式化对比**。

