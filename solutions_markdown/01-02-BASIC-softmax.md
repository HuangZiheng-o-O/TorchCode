[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/02_softmax_solution.ipynb)
# 🟢 Solution: Implement Softmax
Reference solution for the numerically-stable Softmax function.
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
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
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```
```python
# Verify
x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())
print("Ref:   ", torch.softmax(x, dim=-1))
```
```python
# Run judge
from torch_judge import check
check("softmax")
```