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

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]

        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)

        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
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