[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/15_mlp_solution.ipynb)

# 🟠 Solution: SwiGLU MLP

Reference solution — gated feed-forward network used in LLaMA, Mistral, and PaLM.

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
import torch.nn.functional as F
```

```python
# ✅ SOLUTION

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

```python
mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print('Output:', mlp(x).shape)
print('Params:', sum(p.numel() for p in mlp.parameters()))
```

```python
from torch_judge import check
check('mlp')
```