[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/36_int8_quantization_solution.ipynb)
# Solution: INT8 Quantized Linear
Reference solution.
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
```
```python
# ✅ SOLUTION
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer('weight_int8',
            torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8))
        self.register_buffer('scale', scale)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out
```
```python
# Demo
w = torch.randn(8, 4)
q = Int8Linear(w)
print('Output:', q(torch.randn(2, 4)).shape)
print('Weight dtype:', q.weight_int8.dtype)
print('Compression: float32 -> int8 = 4x')
```
```python
from torch_judge import check
check('int8_quantization')
```