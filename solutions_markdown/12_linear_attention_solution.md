[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/12_linear_attention_solution.ipynb)

# 🔴 Solution: Linear Self-Attention

Reference solution — kernel-based attention with elu+1 feature map.

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
import torch.nn.functional as F
```

```python
# ✅ SOLUTION

def linear_attention(Q, K, V):
    Q_prime = F.elu(Q) + 1
    K_prime = F.elu(K) + 1
    KV = torch.bmm(K_prime.transpose(1, 2), V)       # (B, D_k, D_v)
    Z = K_prime.sum(dim=1, keepdim=True)              # (B, 1, D_k)
    num = torch.bmm(Q_prime, KV)                      # (B, S, D_v)
    den = torch.bmm(Q_prime, Z.transpose(1, 2))       # (B, S, 1)
    return num / (den + 1e-6)
```

```python
Q=torch.randn(1,8,16); K=torch.randn(1,8,16); V=torch.randn(1,8,32)
print('Shape:', linear_attention(Q,K,V).shape)
```

```python
from torch_judge import check
check('linear_attention')
```