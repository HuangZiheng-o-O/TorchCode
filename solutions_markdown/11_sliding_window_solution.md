[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/11_sliding_window_solution.ipynb)

# 🔴 Solution: Sliding Window Attention

Reference solution — softmax attention with a band mask.

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

def sliding_window_attention(Q, K, V, window_size):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```

```python
Q=torch.randn(1,6,8); K=torch.randn(1,6,8); V=torch.randn(1,6,8)
print('window=0==V?', torch.allclose(sliding_window_attention(Q,K,V,0), V, atol=1e-5))
```

```python
from torch_judge import check
check('sliding_window')
```