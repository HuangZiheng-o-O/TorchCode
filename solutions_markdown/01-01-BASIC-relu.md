[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/01_relu_solution.ipynb)
# 🟢 Solution: Implement ReLU
Reference solution for the ReLU activation function.
$$\text{ReLU}(x) = \max(0, x)$$
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
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()
```
```python
# Verify
x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
```
```python
# Run judge
from torch_judge import check
check("relu")
```