[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/07_batchnorm_solution.ipynb)
# 🟡 Solution: Implement BatchNorm
Reference solution for Batch Normalization with both **training** and **inference** behavior, including running mean/variance updates.
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
import torch
def my_batch_norm(
    x,
    gamma,
    beta,
    running_mean,
    running_var,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    """BatchNorm with train/eval behavior and running stats.
    - Training: use batch stats, update running_mean / running_var in-place.
    - Inference: use running_mean / running_var as-is.
    """
    if training:
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        # Update running statistics in-place. Detach to avoid tracking gradients.
        running_mean.mul_(1 - momentum).add_(momentum * batch_mean.detach())
        running_var.mul_(1 - momentum).add_(momentum * batch_var.detach())
        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```
```python
# Verify
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
# Training behavior: normalize with batch stats and update running stats
out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("[Train] Column means:", out_train.mean(dim=0))
print("[Train] Column stds: ", out_train.std(dim=0))
print("Updated running_mean:", running_mean)
print("Updated running_var:", running_var)
# Inference behavior: use running_mean / running_var only
out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("[Eval] Column means (using running stats):", out_eval.mean(dim=0))
```
```python
from torch_judge import check
check('batchnorm')
```