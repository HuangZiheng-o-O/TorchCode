# Attention机制四者对比

**Created:** 4/10/2026 22:44:38  
**Updated:** 4/10/2026 22:46:26  
**Exported:** 4/11/2026 0:25:51  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d9b592-85b4-83ea-897c-41716b63531b](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d9b592-85b4-83ea-897c-41716b63531b)  

## Prompt:
4/10/2026, 10:44:45 PM

 **self-attention / masked self-attention / causal attention / cross attention 四者对比**。 think harder

## Response:
4/10/2026, 10:44:48 PM

> Thought for a couple of seconds

四者的核心区别在于：**谁和谁做注意力、是否允许看未来位置、以及 Query/Key/Value 来自哪里**。**Self-attention** 让序列内部任意位置彼此交互；**masked self-attention / causal attention** 本质上是带因果遮罩的 self-attention，只允许当前位置看自己和过去，不能看未来；**cross attention** 则是一个序列用自己的 Query 去“读取”另一个序列提供的 Key 和 Value。面试里最重要的一句话是：**self-attention 解决同一序列内部的信息聚合，causal attention 解决自回归生成的信息泄漏问题，cross attention 解决两个序列或两种表示之间的信息对齐与条件建模。**

The core difference among the four is: **who attends to whom, whether future positions are visible, and where Query/Key/Value come from**. **Self-attention** lets all positions within the same sequence interact with each other; **masked self-attention / causal attention** is essentially self-attention with a causal mask, so each position can only attend to itself and past positions, not future ones; **cross attention** uses the Query from one sequence to “read” the Key and Value from another sequence. The most important interview sentence is: **self-attention handles information aggregation within one sequence, causal attention prevents information leakage in autoregressive generation, and cross attention enables alignment and conditional modeling across two sequences or representations.**

先给统一公式。标准 attention 写成：

The unified formula is:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V
$$

中文解释：这里  $Q$  是 Query，表示“当前 token 想找什么”； $K$  是 Key，表示“每个 token 提供什么索引信号”； $V$  是 Value，表示“真正被聚合的内容”； $d_k$  是 Key 的维度，用  $\sqrt{d_k}$  做缩放是为了防止点积过大导致 softmax 过于尖锐； $M$  是 mask 矩阵，用来控制哪些位置允许被看到，哪些位置必须被屏蔽。

English explanation: here  $Q$  is the Query, representing “what the current token is looking for”;  $K$  is the Key, representing “what indexing signal each token provides”;  $V$  is the Value, representing “the actual content to be aggregated”;  $d_k$  is the key dimension, and dividing by  $\sqrt{d_k}$  prevents the dot product from becoming too large and making softmax too sharp;  $M$  is the mask matrix, which controls which positions are visible and which must be blocked.

* * *

### 1\. Self-attention

**Self-attention 的定义是： $Q,K,V$  都来自同一个输入序列。** 所以它的本质是“同一序列内部做全局信息交互”。每个 token 都可以根据和其他 token 的相似度，从整段序列中动态提取对自己最有用的信息。

**Self-attention is defined by the fact that  $Q,K,V$  all come from the same input sequence.** Its essence is global information interaction within one sequence. Each token dynamically extracts the most useful information for itself from the entire sequence based on similarity to other tokens.

如果输入表示为  $X \in \mathbb{R}^{n \times d}$ ，那么通常有：

If the input representation is  $X \in \mathbb{R}^{n \times d}$ , then typically:

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

中文解释： $X$  是长度为  $n$  的序列表示， $d$  是隐藏维度； $W_Q, W_K, W_V$  是可学习参数矩阵，把同一个输入投影到三个不同的子空间。这样做的作用是让模型把“我要查询什么”“我能提供什么匹配信号”“我真正携带什么内容”拆开建模。

English explanation:  $X$  is the sequence representation of length  $n$ , and  $d$  is the hidden dimension;  $W_Q, W_K, W_V$  are learnable projection matrices that map the same input into three different subspaces. This allows the model to separately represent “what I want to query,” “what matching signal I provide,” and “what actual content I carry.”

在 **encoder** 里，self-attention 通常是**双向的**，也就是一个位置可以看左边和右边所有 token。这很适合理解任务，比如分类、抽取、表示学习，因为模型希望利用完整上下文。

In an **encoder**, self-attention is usually **bidirectional**, meaning each position can see all tokens on both the left and right. This is well suited for understanding tasks such as classification, extraction, and representation learning, because the model benefits from full context.

它的优点是能直接建模长程依赖，而且路径长度是  $O(1)$ ，比 RNN 更容易捕捉远距离关系。它的代价是标准实现的计算和显存复杂度通常是  $O(n^2)$ 。

Its advantage is that it directly models long-range dependencies, with path length  $O(1)$ , making it easier than RNNs to capture distant relationships. The trade-off is that standard implementations usually have  $O(n^2)$  compute and memory complexity.

* * *

### 2\. Masked self-attention

**Masked self-attention 是 self-attention 的一个受限版本： $Q,K,V$  仍然来自同一个序列，但会额外加 mask，禁止某些位置被访问。** 在最常见的语境里，它通常指 decoder 中的“不能看未来”的版本。

**Masked self-attention is a restricted version of self-attention:  $Q,K,V$  still come from the same sequence, but an additional mask is applied to forbid access to certain positions.** In the most common context, it usually refers to the decoder version that cannot look at future tokens.

典型 mask 可以写成：

A typical mask can be written as:

$$
M_{ij} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}
$$

中文解释：这里  $i$  表示当前 query 位置， $j$  表示被关注的 key 位置。当  $j \le i$  时允许访问，所以加 0；当  $j > i$  时代表未来位置，不允许访问，所以加  $-\infty$ ，经过 softmax 后这些位置的概率就变成 0。

English explanation: here  $i$  is the current query position and  $j$  is the key position being attended to. When  $j \le i$ , access is allowed, so we add 0; when  $j > i$ , it represents a future position and is forbidden, so we add  $-\infty$ . After softmax, those positions get probability 0.

所以 masked self-attention 的核心不是换了 attention 机制，而是**在 self-attention 上施加结构约束**。它的目的不是提升表达能力，而是**满足训练目标和推理过程的一致性**。

So the essence of masked self-attention is not a different attention mechanism, but rather **a structural constraint imposed on self-attention**. Its purpose is not to increase expressive power, but to **ensure consistency between the training objective and the inference process**.

* * *

### 3\. Causal attention

**Causal attention 和 masked self-attention 在大多数面试语境里可以视为同义词。** 它强调的是“因果性”——当前位置只能依赖当前和过去信息，不能依赖未来信息。因此它特别适合语言模型这种自回归任务。

**Causal attention and masked self-attention can be treated as synonyms in most interview contexts.** The emphasis is on “causality” — the current position can depend only on current and past information, never future information. That is why it is especially suitable for autoregressive tasks like language modeling.

自回归语言模型训练的是：

An autoregressive language model is trained with:

$$
p(x_1, x_2, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_{<t})
$$

中文解释： $x_t$  表示第  $t$  个 token， $x_{<t}$  表示它之前的所有 token。这个分解说明：生成第  $t$  个词时，模型只能基于过去，而不能偷看未来。所以必须使用 causal mask。

English explanation:  $x_t$  is the token at position  $t$ , and  $x_{<t}$  denotes all previous tokens. This factorization says that when generating the  $t$ \-th token, the model can only rely on the past and cannot peek at the future. Therefore, a causal mask is required.

这里最容易被问到的深入点是：**为什么训练时必须 mask？** 因为如果不 mask，模型在训练时会直接看到目标 token 后面的真实答案，从而产生 information leakage。这样训练损失会虚假地很低，但推理时由于未来 token 不存在，性能会崩掉。

The key deeper interview point here is: **why must we apply the mask during training?** Because without masking, the model would directly see the true future tokens during training, causing information leakage. The training loss would look artificially low, but performance would collapse at inference time because those future tokens are unavailable.

所以更精确地说：**causal attention = 用于满足自回归因果约束的 masked self-attention**。

So more precisely: **causal attention = masked self-attention used to enforce autoregressive causality constraints**.

* * *

### 4\. Cross attention

**Cross attention 的定义是：Query 来自一个序列，而 Key 和 Value 来自另一个序列。** 所以它不是“自己看自己”，而是“一个表示去读取另一个表示”。

**Cross attention is defined by the fact that the Query comes from one sequence, while the Key and Value come from another sequence.** So it is not “self looking at self,” but rather “one representation reading another representation.”

标准形式可以写成：

The standard form can be written as:

$$
Q = X_q W_Q,\quad K = X_{kv} W_K,\quad V = X_{kv} W_V
$$

中文解释： $X_q$  是 query 侧输入， $X_{kv}$  是被读取的上下文输入。比如在 encoder-decoder Transformer 里，decoder 当前隐藏状态产生  $Q$ ，encoder 输出产生  $K,V$ 。这样 decoder 就能在生成当前 token 时，有选择地访问源句子中的相关信息。

English explanation:  $X_q$  is the query-side input, and  $X_{kv}$  is the contextual input being read. For example, in an encoder-decoder Transformer, the decoder’s current hidden states produce  $Q$ , while the encoder outputs produce  $K,V$ . This allows the decoder, when generating the current token, to selectively access relevant information from the source sentence.

cross attention 最经典的应用是机器翻译。decoder 在生成目标语言第  $t$  个词时，先通过 masked self-attention 建模目标侧已有前缀，再通过 cross attention 去对齐源语言句子中最相关的位置。

The classic application of cross attention is machine translation. When the decoder generates the  $t$ \-th target token, it first uses masked self-attention to model the existing target-side prefix, and then uses cross attention to align with the most relevant positions in the source sentence.

它的本质价值是**条件建模**。也就是说，生成或理解一个序列时，不只依赖自身上下文，还依赖另一个外部条件源。这也是为什么多模态模型里文本看图像、图像看文本，本质上也常常通过 cross attention 实现。

Its essential value is **conditional modeling**. That means when generating or understanding one sequence, the model depends not only on its own context but also on another external conditioning source. That is why in multimodal models, text attending to image features or image features attending to text is also often implemented through cross attention.

* * *

### 5\. 四者最本质的对比

**第一，self-attention 和 cross attention 的根本区别是来源不同。** self-attention 中  $Q,K,V$  来自同一序列；cross attention 中  $Q$  和  $K,V$  来自不同序列。

**First, the fundamental difference between self-attention and cross attention is the source of inputs.** In self-attention,  $Q,K,V$  come from the same sequence; in cross attention,  $Q$  and  $K,V$  come from different sequences.

**第二，masked self-attention / causal attention 和普通 self-attention 的根本区别是可见性约束不同。** 普通 self-attention 一般允许看全局；causal attention 只允许看过去。

**Second, the core difference between masked self-attention / causal attention and regular self-attention is the visibility constraint.** Regular self-attention usually allows full visibility; causal attention allows only access to the past.

**第三，masked self-attention 和 causal attention 在绝大多数情况下是同一个东西，只是强调角度不同。** “masked” 强调技术实现上加了 mask；“causal” 强调满足因果顺序、自回归生成约束。

**Third, masked self-attention and causal attention are usually the same thing in practice, but with different emphasis.** “Masked” emphasizes the implementation detail of adding a mask; “causal” emphasizes enforcing temporal causality and autoregressive generation constraints.

* * *

### 6\. 一个非常适合面试复述的表述

可以这样高质量概括：**self-attention 负责同一序列内部的信息融合；masked self-attention 在此基础上加入不可看未来的约束，因此适用于 decoder 的自回归建模；causal attention 本质上就是这种因果 mask 下的 self-attention；cross attention 则让当前序列以 Query 的形式，从另一个序列中选择性读取信息，因此适合 encoder-decoder、多模态、检索增强等条件建模场景。**

A strong interview-ready summary is: **self-attention performs information fusion within the same sequence; masked self-attention adds the constraint that future positions are invisible, making it suitable for autoregressive decoder modeling; causal attention is essentially self-attention under this causal mask; cross attention allows the current sequence, through its Queries, to selectively read information from another sequence, making it suitable for encoder-decoder, multimodal, and retrieval-augmented conditional modeling scenarios.**

* * *

### 7\. Transformer 结构里它们分别出现在哪里

在经典 **encoder-decoder Transformer** 里，encoder 主要使用**双向 self-attention**，因为它要充分理解输入；decoder 先使用**masked self-attention / causal attention**，因为生成当前位置时不能看未来；然后使用**cross attention** 去读取 encoder 输出，从而把源端信息注入到目标端生成过程中。

In the classic **encoder-decoder Transformer**, the encoder mainly uses **bidirectional self-attention** because it needs full understanding of the input; the decoder first uses **masked self-attention / causal attention** because it cannot look at future positions when generating the current token; then it uses **cross attention** to read the encoder outputs, injecting source-side information into target-side generation.

而在 **GPT** 这类纯 decoder 模型里，主要是 **causal self-attention**，通常没有 encoder-decoder 式的 cross attention。相反，在 **T5、BART、原始 Transformer** 这类 seq2seq 模型里，cross attention 是核心组件。

In pure decoder models such as **GPT**, the main mechanism is **causal self-attention**, and they usually do not have encoder-decoder style cross attention. By contrast, in seq2seq models such as **T5, BART, and the original Transformer**, cross attention is a core component.

* * *

### 8\. 为什么这个区别重要

这个区别重要，不只是因为结构不同，而是因为它们分别对应三类不同建模目标：**表示学习、生成约束、条件对齐**。self-attention 强在内部关系建模；causal attention 强在保证训练和推理一致的生成机制；cross attention 强在外部信息注入和对齐。

This distinction matters not just because the structures are different, but because they correspond to three different modeling goals: **representation learning, generation constraints, and conditional alignment**. Self-attention is strong at internal relational modeling; causal attention is strong at ensuring a generation mechanism consistent between training and inference; cross attention is strong at external information injection and alignment.

从更本质的角度讲，attention 不是简单的“加权平均”，而是一个**内容寻址的动态路由机制**。self-attention 是在同一个内存池中检索；cross attention 是在另一个内存池中检索；causal attention 则是在检索时加入时间因果约束。

At a deeper level, attention is not just a “weighted average,” but a **content-addressable dynamic routing mechanism**. Self-attention retrieves from the same memory pool; cross attention retrieves from another memory pool; causal attention adds temporal causality constraints during retrieval.

* * *

### 9\. 最后给一个面试满分式收尾

一句话收尾可以说：**self-attention 看自己全局，masked self-attention / causal attention 看自己过去，cross attention 看别人；前两者解决序列内部依赖建模，后者解决跨序列条件信息交互。**

A strong one-line closing is: **self-attention looks globally at itself, masked self-attention / causal attention looks at its own past, and cross attention looks at another sequence; the first two model dependencies within a sequence, while the last handles conditional information exchange across sequences.**

如果你愿意，我下一条可以继续给你整理成一版 **“面试时 60 秒口述答案”**，或者进一步补上 **四者的矩阵维度、代码实现差异、以及在 GPT/BERT/T5 中的对应关系**。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)