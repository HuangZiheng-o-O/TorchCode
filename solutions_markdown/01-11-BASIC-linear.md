[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/03_linear_solution.ipynb)
# 🟡 Solution: Simple Linear Layer
Reference solution for a fully-connected linear layer: **y = xW^T + b**
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
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```
```python
# Verify
layer = SimpleLinear(8, 4)
print("W shape:", layer.weight.shape)
print("b shape:", layer.bias.shape)
x = torch.randn(2, 8)
print("Output shape:", layer.forward(x).shape)
```
```python
# Run judge
from torch_judge import check
check("linear")
```