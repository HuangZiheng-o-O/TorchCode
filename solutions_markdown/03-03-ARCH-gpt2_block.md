[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/13_gpt2_block_solution.ipynb)
# 🔴 Solution: GPT-2 Transformer Block
Reference solution — pre-norm, causal self-attention, 4x MLP with GELU.
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
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def _attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))
    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```
```python
block = GPT2Block(64, 4)
print('Output:', block(torch.randn(2, 8, 64)).shape)
print('Params:', sum(p.numel() for p in block.parameters()))
```
```python
from torch_judge import check
check('gpt2_block')
```