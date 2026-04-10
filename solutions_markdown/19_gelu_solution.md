[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/19_gelu_solution.ipynb)

# Solution: GELU Activation

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
import math
```

```python
# ✅ SOLUTION

def my_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
```

```python
# Demo
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
```

```python
from torch_judge import check
check('gelu')
```