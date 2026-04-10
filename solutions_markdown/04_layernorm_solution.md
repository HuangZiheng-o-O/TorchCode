[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/04_layernorm_solution.ipynb)

# 🟡 Solution: Implement LayerNorm

Reference solution for Layer Normalization.

$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

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
```

```python
# ✅ SOLUTION

def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```

```python
# Verify
x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))
```

```python
# Run judge
from torch_judge import check
check("layernorm")
```