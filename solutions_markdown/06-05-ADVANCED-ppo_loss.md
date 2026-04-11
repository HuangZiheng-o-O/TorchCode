[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/39_ppo_loss_solution.ipynb)
# Solution: PPO Clipped Loss
Reference solution for the PPO clipped surrogate loss task.
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
from torch import Tensor
```
```python
# ✅ SOLUTION
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    """PPO clipped surrogate loss.
    new_logps: (B,) current policy log-probs
    old_logps: (B,) old policy log-probs (treated as constant)
    advantages: (B,) advantage estimates (treated as constant)
    returns: scalar loss (Tensor)
    """
    # Detach old_logps and advantages so gradients only flow through new_logps
    old_logps_detached = old_logps.detach()
    adv_detached = advantages.detach()
    # Importance sampling ratio r = pi_new / pi_old in log-space
    ratios = torch.exp(new_logps - old_logps_detached)
    # Unclipped and clipped objectives
    unclipped = ratios * adv_detached
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached
    # PPO objective: negative mean of the more conservative objective
    return -torch.min(unclipped, clipped).mean()
```
```python
# Demo
new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
print('Loss:', ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2))
```
```python
from torch_judge import check
check('ppo_loss')
```