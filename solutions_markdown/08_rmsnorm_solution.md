[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/08_rmsnorm_solution.ipynb)

# 🟡 Solution: Implement RMSNorm

Reference solution for Root Mean Square Normalization.

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

def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```

```python
x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())
```

```python
from torch_judge import check
check('rmsnorm')
```