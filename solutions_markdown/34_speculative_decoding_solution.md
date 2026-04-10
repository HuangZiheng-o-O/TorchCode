[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/34_speculative_decoding_solution.ipynb)

# Solution: Speculative Decoding

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
```

```python
# ✅ SOLUTION

def speculative_decode(target_probs, draft_probs, draft_tokens):
    K = len(draft_tokens)
    accepted = []
    for i in range(K):
        t = draft_tokens[i].item()
        ratio = target_probs[i, t] / max(draft_probs[i, t].item(), 1e-10)
        if torch.rand(1).item() < min(1.0, ratio.item()):
            accepted.append(t)
        else:
            adjusted = torch.clamp(target_probs[i] - draft_probs[i], min=0)
            s = adjusted.sum()
            if s > 0:
                adjusted = adjusted / s
            else:
                adjusted = torch.ones_like(adjusted) / adjusted.shape[0]
            accepted.append(torch.multinomial(adjusted, 1).item())
            return accepted
    return accepted
```

```python
# Demo
torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
print('Perfect draft:', speculative_decode(probs, probs, tokens))
```

```python
from torch_judge import check
check('speculative_decoding')
```