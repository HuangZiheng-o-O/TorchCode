[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/06_multihead_attention_solution.ipynb)
# 🔴 Solution: Multi-Head Attention
Reference solution for the Multi-Head Attention mechanism.
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
```python
# Install torch-judge in Colab (no-op in JupyterLab/Docker)
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
# ✅ SOLUTION
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)  # d_model = num_heads × d_k

    def forward(self, Q, K, V, mask=None):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]  # K: (B, S_k, d_model)
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        # (B, num_heads, S_q, d_k) @ (B, num_heads, d_k, S_k)
        # → (B, num_heads, S_q, S_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores.shape = (B, num_heads, S_q, S_k)
        if mask is not None:# 当 mask == 0 时，把对应位置的 scores 填成 -inf 
            scores = scores.masked_fill(mask == 0, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        # weights: (B, num_heads, S_q, S_k)
        # v: (B, num_heads, S_k, d_k)
        attn = torch.matmul(weights, v)
        # attn: (B, num_heads, S_q, d_k)
        # transpose to #(B, S_q, num_heads, d_k)
        out = (
            attn.transpose(1, 2)
                .contiguous()  # transpose 后，tensor 在内存中是“非连续”的
                .view(B, S_q, -1)
        )
        return self.W_o(out)
```
```python
# Verify
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Self-attn shape:", out.shape)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)
```
```python
# Run judge
from torch_judge import check
check("mha")
```


```python
# 对于固定的一个 (batch i, head h, query q)

scores[i, h, q, :]   # shape: (S_k,)
# 表示：这个 query 对所有 key 的打分

weights = softmax(scores, dim=-1)

weights[i, h, q, :]  # shape: (S_k,)
# 表示：这个 query 对所有 key 的注意力权重（概率分布）

# 性质：
sum(weights[i, h, q, :]) == 1

``` 