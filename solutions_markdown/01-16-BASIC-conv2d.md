[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/22_conv2d_solution.ipynb)
# Solution: 2D Convolution
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
import torch.nn.functional as F
```
```python
# ✅ SOLUTION
def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        x = F.pad(x, [padding] * 4)
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
    out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
```
```python
# Demo
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print('Output:', my_conv2d(x, w).shape)
print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
```
```python
from torch_judge import check
check('conv2d')
```