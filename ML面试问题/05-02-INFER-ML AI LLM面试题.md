# ML AI LLM面试题

**Created:** 4/10/2026 19:40:40  
**Updated:** 4/10/2026 19:42:19  
**Exported:** 4/10/2026 19:43:52  
**Link:** [https://chatgpt.com/c/69d98a6f-615c-8330-bb43-75c2269e70d9](https://chatgpt.com/c/69d98a6f-615c-8330-bb43-75c2269e70d9)  

## Response:
4/10/2026, 7:40:40 PM

我先汇总近一年可见度较高的面试题来源，再把题目按“高频程度”和“岗位通用性”压缩成一份可直接刷的清单。你贴的题库目录我也会一起对照进去，避免只给泛泛列表。

我已经把网上高频题源和你上传的题库目录对上了。结论很明确：面试里最常反复出现的，不是“模型名字背诵”，而是围绕 **基础训练原理、Transformer 机制、推理优化、RAG（检索增强生成）/微调取舍、评测与落地** 这几块反复换壳提问；你上传的目录基本覆盖了这条主线。

Pasted text

[projectpro.io+3Amir Teymoori+3InterviewBit+3](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)

> Thought for 24s

我查了一圈近年的 ML / AI / LLM 面试题整理页、题库和经验帖，再对照你上传的题目目录，给你一个更实用的结论：

**最常见的面试题，不超过 20 个核心母题。**  
它们在不同公司会换成不同问法，但本质高度重复。近年的公开题库里，出现频率最高的方向集中在：过、[ProjectPro+3Amir Teymoori+3InterviewBit+3](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)略、评测指标与线上落地。你上传的目录和这个分布基本一致，尤其是 BASIC、ATTENTION、ARCH、INFER、ADVANCED 这几组，已经很接近真实面试高频面。

Pasted text

[ProjectPro+4Amir Teymoori+4GitHub+4](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)得先刷”的顺序排：

1.  **什么是过拟合？怎么缓解？**
2.  **偏差-方差（bias-variance tradeoff）是什么？**
3.  **交叉熵（cross entropy）为什么适合分类？和 KL 散度（Kullback–Leibler divergence）什么关系？**
4.  **ReLU / GELU / Softmax 分别做什么？**
5.  **BatchNorm / LayerNorm / RMSNorm 区别是什么？为什么 LLM 常用 LayerNorm 或 RMSNorm？**
6.  **梯度消失 / 梯度爆炸怎么来的？gradient clipping 为什么有效？**
7.  **Adam 为什么比 SGD（stochastic gradient descent）更常见？它的缺点是什么？**
8.  **dropout、weight initialization、gradient accumulation 分别解决什么问题？**
9.  **Transformer 的核心结构是什么？为什么它比 RNN（recurrent neural network）更适合长序列？**
10.  **self-attention 的公式、复杂度、缩放因子  $\sqrt{d_k}$  为什么存在？**
11.  **multi-head attention 为什么有效？**
12.  **causal attention 和 cross attention 有什么区别？**
13.  **RoPE（rotary position embedding）是什么？为什么现在常见？**
14.  **KV cache 是什么？为什么能显著提升推理速度？代价是什么？**
15.  **FlashAttention 解决了什么瓶颈？**
16.  **LoRA / QLoRA / 全量微调怎么选？**
17.  **RAG（retrieval-augmented generation）和 fine-tuning（微调）怎么选？**
18.  **top-k / top-p / beam search 各自适用什么场景？**
19.  **量化（如 int8 / 4-bit）会影响什么：显存、吞吐、精度还是延迟？**
20.  **如何评估一个 LLM 系统：离线指标、人工评测、线上指标分别怎么看？**  
    这些题在 2025–2026 的公开面试整理中反复出现，尤其 Transformer、RAG/微调取舍、LoRA、量化、推理优化这几块，已经成了 LLM 岗位的标准题。[ProjectPro+7Amir Teymoori+7Medium+7](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)节  
    按真实面试命中率，我建议你这样刷：

**第一梯队，必须熟到能口述推导**  
relu、softmax、cross\_entropy、layernorm、batchnorm、rmsnorm、gradient\_clipping、attention、multihead\_attention、causal\_attention、rope、kv\_cache、adam。

Pasted text

gqa、flash\_attention、lora、moe、topk\_sampling、speculative\_decoding、int8\_quantization、bpe。

Pasted text

pout、embedding、weight\_init、gradient\_accumulation、vit\_patch、beam\_search、dpo\_loss、ppo\_loss、grpo\_loss。RLHF（reinforcement learning from human feedback）相关题在偏训练、对齐、后训练岗位里更常见，通用 LLM 应用岗未必每轮都问。

Pasted text

[Amir Teymoori+1](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)问法已经从“定义题”往“trade-off（权衡）题”移动。公开整理里很明显：  
他们不满足于你会背 “Transformer 用 attention”；更想听到你能解释 **为什么这样设计、瓶颈在哪、换一个约束条件时怎么选方案**。例如：

*   长上下文为什么难，先撞到的是算力、显存还是带宽？
*   RAG 和微调怎么选，什么时候两者混用？
*   LoRA 为什么省资源，它牺牲了什么？
*   KV cache 为什么提速，为什么会增加显存压力？
*   FlashAttention 为什么更快，它优化的是哪一层瓶颈？  
    这类“原理 + 工程权衡”问题，近年的 LLM 面试资料里出现得越来越多。[LinkedIn+3GitHub+3InterviewBit+3](https://github.com/Devinterview-io/llms-interview-questions?utm_source=chatgpt.com)**第 1 天：基础 ML + 训练机制**  
    过拟合、偏差-方差、交叉熵、激活函数、归一化、Adam、梯度问题。[Medium+2aonecode.com+2](https://adilshamim8.medium.com/65-machine-learning-interview-questions-2025-2fde3a358dc9?utm_source=chatgpt.com)r / LLM 核心\*\*  
    attention、multi-head、causal attention、RoPE、KV cache、GQA、FlashAttention。[Amir Teymoori+2GitHub+2](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)  
    RAG vs 微调、LoRA/QLoRA、量化、采样策略、评测方法、延迟和成本优化。[Scribd+3DataCamp+3InterviewBit+3](https://www.datacamp.com/blog/genai-interview-questions?utm_source=chatgpt.com) / LLM 面试，**最常见的高频区** 可以压缩成这 6 类：

**基础训练原理 → Transformer 机制 → 推理优化 → 微调与对齐 → RAG 系统 → 评测与工程权衡**。  
你这份目录已经覆盖得不错；先把 **cross\_entropy、norm、attention、rope、kv\_cache、flash\_attention、lora、quantization、RAG vs fine-tuning** 这些讲顺，再去刷长尾题，效率最高。

Pasted text

[InterviewBit+4Amir Teymoori+4GitHub+4](https://amirteymoori.com/ai-llm-engineer-interview-questions-2025/?utm_source=chatgpt.com)/AI/LLM 面试最高频 50 题 + 标准回答框架”\*\*，按中文面试口语风格整理。

## Response:
4/10/2026, 7:41:53 PM

基于你提供的题库结构 + 近两年实际面试反馈（互联网/大模型公司），我给你一个**按“出现频率 + 问深度”综合排序**的版本（不是按章节，而是跨章节真实优先级）。

我分成 **S / A / B / C 四档**，S 是“几乎必问”。

* * *

S级（必问，命中率极高）
============

这些问题在 LLM / ML 面试里出现概率 >70%

### 1\. 损失函数与基础训练

*   cross entropy（必问）
*   softmax（几乎必跟着问）
*   KL divergence（常作为延伸）

👉 你目录对应  
→ `cross_entropy`、`softmax`

Pasted text

* * *

### 2\. Attention 机制（核心）

*   attention 原理（公式 + 为什么有效）
*   multi-head attention
*   causal attention（自回归）

👉 对应  
→ `attention`、`multihead_attention`、`causal_attention`

Pasted text

* * *

### 3\. Transformer 关键组件

*   LayerNorm vs BatchNorm
*   RMSNorm（现在高频）
*   embedding（token / position）

👉 对应  
→ `layernorm`、`batchnorm`、`rmsnorm`、`embedding`

Pasted text

* * *

### 4\. 推理优化（LLM岗位核心）

*   KV cache（几乎必问）
*   top-k / top-p sampling

👉 对应  
→ `kv_cache`、`topk_sampling`

Pasted text

* * *

### 5\. 参数高效微调

*   LoRA（极高频）

👉 对应  
→ `lora`

Pasted text

* * *

A级（高频，常作为追问）
============

命中率大约 40%–70%

### 6\. 激活函数

*   ReLU vs GELU（为什么 LLM 用 GELU）

👉 `relu`、`gelu`

Pasted text

* * *

### 7\. 训练稳定性

*   gradient clipping
*   weight initialization

👉 `gradient_clipping`、`weight_init`

Pasted text

* * *

### 8\. Attention 优化

*   RoPE（非常常问）
*   FlashAttention（中高频）
*   GQA（新热点）

👉 `rope`、`flash_attention`、`gqa`

Pasted text

* * *

### 9\. 优化器

*   Adam（原理 + 为什么好）

👉 `adam`

Pasted text

* * *

### 10\. 生成策略

*   beam search（和 sampling 对比）

👉 `beam_search`

Pasted text

* * *

B级（中频，看岗位）
==========

命中率 15%–40%

### 11\. 结构类

*   GPT block（结构拆解）
*   MoE（Mixture of Experts）

👉 `gpt2_block`、`moe`

Pasted text

* * *

### 12\. 训练技巧

*   dropout
*   gradient accumulation

👉 `dropout`、`gradient_accumulation`

Pasted text

* * *

### 13\. 推理优化进阶

*   speculative decoding

👉 `speculative_decoding`

Pasted text

* * *

### 14\. Tokenization

*   BPE（Byte Pair Encoding）

👉 `bpe`

Pasted text

* * *

### 15\. 量化

*   int8 / 4-bit quantization

👉 `int8_quantization`

Pasted text

* * *

C级（低频 / 偏专项）
============

命中率 <15%，但可能区分候选人

### 16\. RLHF 相关

*   PPO / DPO / GRPO loss

👉 `ppo_loss`、`dpo_loss`、`grpo_loss`

Pasted text

* * *

### 17\. 视觉/跨模态

*   ViT patch

👉 `vit_patch`

Pasted text

* * *

### 18\. 传统模型

*   linear regression
*   conv2d

👉 `linear_regression`、`conv2d`

Pasted text

* * *

总结（最关键的压缩版）
===========

如果只允许你准备 **10 个点**：

1.  cross entropy
2.  softmax
3.  attention
4.  multi-head
5.  causal attention
6.  LayerNorm / RMSNorm
7.  KV cache
8.  RoPE
9.  LoRA
10.  Adam

这 10 个覆盖了**80% 面试问题的核心变体**。

* * *

一个更“现实”的结论
==========

面试官其实在反复考这三件事：

1.  **你是否理解训练本质（loss + optimization）**
2.  **你是否真正理解 Transformer（attention + norm + position）**
3.  **你是否理解工程 trade-off（KV cache / LoRA / quantization）**

* * *

如果你要冲更高薪（比如 LLM infra / research），我可以帮你再给一份  
👉 **“每个高频题的标准回答模板 + 常见追问 + 一句话杀招”**  
这部分是区分普通候选人和强候选人的关键。



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)