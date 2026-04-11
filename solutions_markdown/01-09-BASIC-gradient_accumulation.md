[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/31_gradient_accumulation_solution.ipynb)
# Solution: Gradient Accumulation
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
def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss
```
```python
# Demo
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)
```
```python
from torch_judge import check
check('gradient_accumulation')
```