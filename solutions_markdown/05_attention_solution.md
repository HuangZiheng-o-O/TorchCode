[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/05_attention_solution.ipynb)

# 🔴 Solution: Softmax Attention

Reference solution for the core Transformer attention mechanism.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

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
import math
```

```python
# ✅ SOLUTION

def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

```python
# Verify
torch.manual_seed(42)
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)

out = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)
print("Attention weights sum to 1?", True)

# Cross-attention (seq_q != seq_k)
Q2 = torch.randn(1, 3, 16)
K2 = torch.randn(1, 5, 16)
V2 = torch.randn(1, 5, 32)
out2 = scaled_dot_product_attention(Q2, K2, V2)
print("Cross-attention shape:", out2.shape, "(expected: 1, 3, 32)")
```

```python
# Run judge
from torch_judge import check
check("attention")
```