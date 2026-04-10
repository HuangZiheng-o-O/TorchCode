[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/37_dpo_loss_solution.ipynb)

# Solution: DPO (Direct Preference Optimization) Loss

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

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

```python
# Demo
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
```

```python
from torch_judge import check
check('dpo_loss')
```