[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/38_grpo_loss_solution.ipynb)

# Solution: GRPO (Group Relative Policy Optimization) Loss

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
from torch import Tensor
```

```python
# ✅ SOLUTION

def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    """Group Relative Policy Optimization (GRPO) loss.

    logps: (B,) policy log-probs for each sampled response
    rewards: (B,) scalar rewards for each response
    group_ids: (B,) integers, same id = same prompt/group
    returns: scalar loss (Tensor)
    """
    # Compute per-group normalized advantages A_i
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)

    # Stop gradient through advantages
    advantages_detached = advantages.detach()

    # GRPO objective: -E[A_i * logpi_i]
    return -(advantages_detached * logps).mean()

```

```python
# Demo
logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print('Loss:', grpo_loss(logps, rewards, group_ids).item())
```

```python
from torch_judge import check
check('grpo_loss')
```