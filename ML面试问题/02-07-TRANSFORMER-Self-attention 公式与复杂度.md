# Self-attention 公式与复杂度

**Created:** 4/10/2026 21:05:28  
**Updated:** 4/10/2026 21:06:52  
**Exported:** 4/10/2026 21:08:38  
**Link:** [https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99e57-fdf4-83ea-b99f-b295d5f23024](https://chatgpt.com/g/g-69d92499d7488191ba8c4ab3fd6255f3-ml-mian-shi-hui-da-zhuan-jia/c/69d99e57-fdf4-83ea-b99f-b295d5f23024)  

## Prompt:
4/10/2026, 9:05:28 PM

        
self-attention 的公式、复杂度、缩放因子 \sqrt{d_k} 为什么存在？ think harder

## Response:
4/10/2026, 9:05:29 PM

> Thought for a couple of seconds

Self-attention 的本质是：**对每个 query，用它和所有 key 的相似度去生成一组权重，再用这组权重对 value 做加权求和**。它的核心公式是

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

其中， $Q\in\mathbb{R}^{n_q\times d_k}$  是 query 矩阵， $K\in\mathbb{R}^{n_k\times d_k}$  是 key 矩阵， $V\in\mathbb{R}^{n_k\times d_v}$  是 value 矩阵， $d_k$  是 key/query 的特征维度。 $QK^\top$  产生两两 token 间的相似度分数，softmax 把分数变成权重分布，再对  $V$  做加权求和得到输出。  
The essence of self-attention is: **for each query, compute similarities with all keys, turn them into weights, and then take a weighted sum of the values**. Its core formula is

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

where  $Q\in\mathbb{R}^{n_q\times d_k}$  is the query matrix,  $K\in\mathbb{R}^{n_k\times d_k}$  is the key matrix,  $V\in\mathbb{R}^{n_k\times d_v}$  is the value matrix, and  $d_k$  is the query/key feature dimension.  $QK^\top$  gives pairwise token similarity scores, softmax converts those scores into a weight distribution, and the weighted sum over  $V$  produces the output.

在 self-attention 里， $Q,K,V$  通常来自**同一个输入序列**的线性投影。设输入为  $X\in\mathbb{R}^{n\times d_{\text{model}}}$ ，则

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

其中， $W_Q\in\mathbb{R}^{d_{\text{model}}\times d_k}$ 、 $W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$ 、 $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$  是可学习参数矩阵。它们的作用是把同一个 token 投影到三个不同语义空间：query 表示“我要找什么”，key 表示“我能提供什么线索”，value 表示“真正被聚合的内容”。  
In self-attention,  $Q,K,V$  usually come from linear projections of the **same input sequence**. Let the input be  $X\in\mathbb{R}^{n\times d_{\text{model}}}$ . Then

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

where  $W_Q\in\mathbb{R}^{d_{\text{model}}\times d_k}$ ,  $W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$ , and  $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$  are learnable parameter matrices. Their role is to project the same token into three different semantic spaces: the query means “what I am looking for,” the key means “what evidence I can provide,” and the value means “what content will actually be aggregated.”

如果看单个位置  $i$  对所有位置的注意力，它可以写成更直观的形式：

$$
\alpha_{ij}=\frac{\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)}{\sum_{t=1}^{n}\exp\left(\frac{q_i^\top k_t}{\sqrt{d_k}}\right)}
$$
 
$$
o_i=\sum_{j=1}^{n}\alpha_{ij}v_j
$$

这里， $q_i$  是第  $i$  个位置的 query， $k_j$  是第  $j$  个位置的 key， $v_j$  是第  $j$  个位置的 value， $\alpha_{ij}$  表示位置  $i$  对位置  $j$  的注意力权重， $o_i$  是最终输出。这个公式说明 self-attention 不是“选一个最相关 token”，而是“形成一个上下文相关的加权组合”。  
If we look at the attention from a single position  $i$  to all positions, it can be written more intuitively as

$$
\alpha_{ij}=\frac{\exp\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right)}{\sum_{t=1}^{n}\exp\left(\frac{q_i^\top k_t}{\sqrt{d_k}}\right)}
$$
 
$$
o_i=\sum_{j=1}^{n}\alpha_{ij}v_j
$$

Here,  $q_i$  is the query at position  $i$ ,  $k_j$  is the key at position  $j$ ,  $v_j$  is the value at position  $j$ ,  $\alpha_{ij}$  is the attention weight from position  $i$  to position  $j$ , and  $o_i$  is the final output. This shows that self-attention does not merely “pick the single most relevant token”; it forms a context-dependent weighted combination.

复杂度上，self-attention 最核心的开销来自  $QK^\top$  和 attention weights 再乘  $V$ 。如果序列长度是  $n$ ，隐藏维度近似记为  $d$ ，那么

$$
QK^\top:\ \mathcal{O}(n^2d),\qquad \mathrm{softmax}(A)V:\ \mathcal{O}(n^2d)
$$

所以总时间复杂度通常写成

$$
\mathcal{O}(n^2d)
$$

空间复杂度里最关键的是 attention score 矩阵  $A=QK^\top\in\mathbb{R}^{n\times n}$ ，因此主导项是

$$
\mathcal{O}(n^2)
$$

这也是为什么长序列是 self-attention 的主要瓶颈：它不是主要被参数量卡住，而是被 token 两两交互的二次复杂度卡住。  
In terms of complexity, the dominant cost in self-attention comes from  $QK^\top$  and then multiplying the attention weights by  $V$ . If the sequence length is  $n$  and the hidden dimension is approximately  $d$ , then

$$
QK^\top:\ \mathcal{O}(n^2d),\qquad \mathrm{softmax}(A)V:\ \mathcal{O}(n^2d)
$$

So the overall time complexity is usually written as

$$
\mathcal{O}(n^2d)
$$

For space complexity, the most important object is the attention score matrix  $A=QK^\top\in\mathbb{R}^{n\times n}$ , so the dominant term is

$$
\mathcal{O}(n^2)
$$

This is why long sequences are the main bottleneck for self-attention: the issue is not primarily the parameter count, but the quadratic all-pairs token interaction.

更细一点，如果是 multi-head attention，设 head 数为  $h$ ，每个 head 的 key 维度是  $d_k=d_{\text{model}}/h$ ，那么**总复杂度量级并不会因为多头而变成  $h$  倍更大**。因为每个 head 的维度变小了，单头代价约为  $\mathcal{O}(n^2 d_k)$ ， $h$  个 head 加起来仍大致是

$$
\mathcal{O}(n^2 d_{\text{model}})
$$

多头带来的主要收益不是降低复杂度，而是让模型能在不同子空间里学习不同类型的关系，比如局部句法关系、长程依赖、指代关系等。  
More precisely, in multi-head attention, suppose there are  $h$  heads and each head has key dimension  $d_k=d_{\text{model}}/h$ . Then the **overall order of complexity does not become  $h$  times larger** just because of multiple heads. Since each head has smaller dimensionality, the cost per head is about  $\mathcal{O}(n^2 d_k)$ , and summing over  $h$  heads still gives roughly

$$
\mathcal{O}(n^2 d_{\text{model}})
$$

The main benefit of multi-head attention is not lower complexity, but the ability to learn different types of relationships in different subspaces, such as local syntax, long-range dependency, and coreference.

 $\sqrt{d_k}$  这个缩放因子的存在，本质上是为了**控制点积  $q_i^\top k_j$  的数值尺度，避免随着维度增大，softmax 过早进入饱和区间**。最经典的解释来自方差分析。假设  $q_i$  和  $k_j$  的每个分量都是独立、零均值、单位方差，那么

$$
q_i^\top k_j=\sum_{\ell=1}^{d_k} q_{i\ell}k_{j\ell}
$$

因为每一项  $q_{i\ell}k_{j\ell}$  的均值约为 0，方差约为 1，所以总和的方差近似为

$$
\mathrm{Var}(q_i^\top k_j)\approx d_k
$$

于是其标准差约为

$$
\sqrt{d_k}
$$

因此把点积除以  $\sqrt{d_k}$ ，就能把 logits 的尺度重新拉回到大致方差为 1 的范围。  
The scaling factor  $\sqrt{d_k}$  exists essentially to **control the numerical scale of the dot product  $q_i^\top k_j$ , preventing softmax from entering saturation too early as the dimension grows**. The classical explanation comes from variance analysis. Assume each component of  $q_i$  and  $k_j$  is independent, zero-mean, and unit-variance. Then

$$
q_i^\top k_j=\sum_{\ell=1}^{d_k} q_{i\ell}k_{j\ell}
$$

Since each term  $q_{i\ell}k_{j\ell}$  has mean approximately 0 and variance approximately 1, the variance of the sum is roughly

$$
\mathrm{Var}(q_i^\top k_j)\approx d_k
$$

So its standard deviation is about

$$
\sqrt{d_k}
$$

Therefore, dividing the dot product by  $\sqrt{d_k}$  rescales the logits back to a range with variance roughly 1.

为什么 softmax 饱和是个问题？因为 softmax 对输入尺度非常敏感。假设某一行 attention logits 很大，比如一个是 12，另一个是 3，那么 softmax 会极度偏向较大的那个值，输出接近 one-hot。这样会带来两个后果：第一，模型在训练初期就可能过度自信，注意力分布太尖锐；第二，梯度会变小，训练不稳定或者收敛变慢。  
Why is softmax saturation a problem? Because softmax is very sensitive to input scale. Suppose one row of attention logits is large, for example one value is 12 and another is 3. Then softmax will strongly favor the larger one and produce an output close to one-hot. This leads to two consequences: first, the model may become overconfident too early in training, making the attention distribution too sharp; second, gradients become small, which can make training unstable or slower to converge.

从梯度角度看更深一层，softmax 的导数是

$$
\frac{\partial\, \mathrm{softmax}(z)_i}{\partial z_j} =\mathrm{softmax}(z)_i\left(\delta_{ij}-\mathrm{softmax}(z)_j\right)
$$

其中， $z$  是 logits 向量， $\delta_{ij}$  是 Kronecker delta，当  $i=j$  时为 1，否则为 0。这个公式说明：一旦某个分量的 softmax 概率接近 1，其对应导数和其他位置的导数都会接近 0，梯度传播就会变差。缩放  $\sqrt{d_k}$  的作用，本质上是在 attention logits 进入 softmax 之前做一个“温度校准”，避免过早极化。  
From the gradient perspective, the softmax derivative is

$$
\frac{\partial\, \mathrm{softmax}(z)_i}{\partial z_j} =\mathrm{softmax}(z)_i\left(\delta_{ij}-\mathrm{softmax}(z)_j\right)
$$

where  $z$  is the logits vector and  $\delta_{ij}$  is the Kronecker delta, equal to 1 when  $i=j$  and 0 otherwise. This formula shows that once one softmax probability becomes close to 1, both its own derivative and the derivatives for other positions approach 0, so gradient flow deteriorates. The role of dividing by  $\sqrt{d_k}$  is essentially a “temperature calibration” before applying softmax, preventing premature polarization.

你可以把它理解成：**点积相似度会天然随着维度增长而变大，但这种变大并不一定意味着“更相关”，很多时候只是高维累加造成的数值膨胀**。所以  $\sqrt{d_k}$  不是随便拍脑袋加的技巧，而是一个和高维随机向量统计规律匹配的归一化。  
You can interpret it this way: **dot-product similarity naturally grows with dimension, but this growth does not necessarily mean “more relevant”; often it is just numerical inflation caused by high-dimensional summation**. So  $\sqrt{d_k}$  is not an arbitrary trick; it is a normalization matched to the statistical behavior of high-dimensional random vectors.

再往深一点说，为什么不是除以  $d_k$ ，而是除以  $\sqrt{d_k}$ ？因为点积的**方差**随  $d_k$  线性增长，而标准差随  $d_k$  的平方根增长。我们通常要对齐的是“典型波动尺度”，也就是标准差，而不是方差本身。所以合理的归一化量级是  $\sqrt{d_k}$ ，不是  $d_k$ 。如果除以  $d_k$ ，logits 会被压得过小，attention 过于平，模型区分能力会下降。  
Going one level deeper, why divide by  $\sqrt{d_k}$  rather than by  $d_k$ ? Because the **variance** of the dot product grows linearly with  $d_k$ , while the standard deviation grows with the square root of  $d_k$ . What we usually want to normalize is the “typical fluctuation scale,” namely the standard deviation, not the variance itself. So the correct normalization order is  $\sqrt{d_k}$ , not  $d_k$ . If we divided by  $d_k$ , the logits would become too small, the attention would become overly flat, and the model’s discriminative power would weaken.

还有一个面试里很加分的理解是： $\sqrt{d_k}$  可以看作把 dot-product attention 变成“尺度更稳定的相似度函数”。因为 attention 的核心不是单纯比较两个向量的绝对点积，而是需要一个**在不同维度设置下都可训练、可比较、数值稳定**的打分机制。没有这个缩放，模型维度一改，logit 分布就会显著变，训练行为也会变。  
A strong interview-level interpretation is that  $\sqrt{d_k}$  makes dot-product attention a **more scale-stable similarity function**. The goal of attention is not merely to compare absolute dot products, but to use a scoring mechanism that remains **trainable, comparable, and numerically stable across different dimensional settings**. Without this scaling, changing the model dimension would significantly alter the logit distribution and therefore the training dynamics.

当然，严格来说， $\mathrm{Var}(q_i^\top k_j)\approx d_k$  依赖一些简化假设，比如分量独立、均值接近 0、方差接近 1。真实训练后的  $Q,K$  不会完全满足这些条件。但这个结论仍然足够说明主导趋势：**维度变大时，未缩放点积的数值尺度会系统性变大**，因此需要标准化。  
Strictly speaking, the statement  $\mathrm{Var}(q_i^\top k_j)\approx d_k$  relies on simplifying assumptions such as independent components, near-zero means, and near-unit variances. In a trained model, real  $Q$  and  $K$  do not perfectly satisfy these assumptions. Still, the conclusion captures the dominant trend well: **as dimensionality increases, the scale of unnormalized dot products systematically grows**, so normalization is needed.

把它和 cosine similarity 对比也很有帮助。cosine similarity 会显式除以向量范数：

$$
\cos(\theta)=\frac{q^\top k}{\|q\|\|k\|}
$$

其中， $q^\top k$  是点积， $\|q\|$  和  $\|k\|$  是向量范数。它的作用是去掉长度影响，只保留方向相似性。attention 没有直接用 cosine similarity，而是用缩放点积，这是因为缩放点积计算更高效，也更容易和线性投影、矩阵乘法流水线结合。但它们背后的共同目标都是：**避免相似度分数因尺度问题失真**。  
It is also helpful to compare this with cosine similarity, which explicitly divides by vector norms:

$$
\cos(\theta)=\frac{q^\top k}{\|q\|\|k\|}
$$

where  $q^\top k$  is the dot product and  $\|q\|$ ,  $\|k\|$  are the vector norms. Its role is to remove magnitude effects and preserve only directional similarity. Attention does not directly use cosine similarity; instead it uses scaled dot product, mainly because scaled dot product is computationally more efficient and fits naturally into linear projection and matrix multiplication pipelines. But the shared purpose is the same: **prevent similarity scores from being distorted by scale**.

最后用一句面试式总结：**self-attention 就是“基于内容寻址的动态加权聚合”机制，复杂度瓶颈是 token 两两交互带来的  $\mathcal{O}(n^2)$ ，而  $\sqrt{d_k}$  的作用是把点积 logits 的尺度控制住，防止 softmax 饱和，从而保持训练稳定和梯度健康。**  
To summarize in an interview-ready sentence: **self-attention is a content-based dynamic weighted aggregation mechanism; its complexity bottleneck comes from the  $\mathcal{O}(n^2)$  all-pairs token interaction, and the purpose of  $\sqrt{d_k}$  is to control the scale of dot-product logits, prevent softmax saturation, and thus preserve training stability and healthy gradients.**

你在面试里如果想答到更强，可以补这一句：**“缩放因子不是为了改变表达能力，而是为了改善优化性质；它主要影响的是训练时的数值稳定性和梯度行为。”**  
If you want to make the answer even stronger in an interview, add this sentence: **“The scaling factor is not there to change representational power; it is there to improve optimization behavior, mainly affecting numerical stability and gradient dynamics during training.”**



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)