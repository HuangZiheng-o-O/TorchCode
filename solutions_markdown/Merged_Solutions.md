# 🟢 Solution: Implement ReLU
Reference solution for the ReLU activation function.
$$\text{ReLU}(x) = \max(0, x)$$
```python
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
def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()
```
```python
x = torch.tensor([-2., -1., 0., 1., 2.])
print("Input: ", x)
print("Output:", relu(x))
```
```python
```
# 🟢 Solution: Implement Softmax
Reference solution for the numerically-stable Softmax function.
$$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$
```python
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
def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)
```
```python
x = torch.tensor([1.0, 2.0, 3.0])
print("Output:", my_softmax(x, dim=-1))
print("Sum:   ", my_softmax(x, dim=-1).sum())
print("Ref:   ", torch.softmax(x, dim=-1))
```
```python
```
# Solution: Cross-Entropy Loss
Reference solution.
```python
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
def cross_entropy_loss(logits, targets):
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    return -log_probs[torch.arange(targets.shape[0]), targets].mean()
```
```python
logits = torch.randn(4, 10)
targets = torch.randint(0, 10, (4,))
print('Loss:', cross_entropy_loss(logits, targets).item())
print('Ref: ', torch.nn.functional.cross_entropy(logits, targets).item())
```
```python
```
# Solution: Implement Dropout
Reference solution.
```python
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
class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        mask = (torch.rand_like(x) > self.p).float()
        return x * mask / (1 - self.p)
```
```python
d = MyDropout(p=0.5)
d.train()
x = torch.ones(10)
print('Train:', d(x))
d.eval()
print('Eval: ', d(x))
```
```python
```
# Solution: Embedding Layer
Reference solution.
```python
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
class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))
    def forward(self, indices):
        return self.weight[indices]
```
```python
emb = MyEmbedding(10, 4)
idx = torch.tensor([0, 3, 7])
print('Output shape:', emb(idx).shape)
print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))
```
```python
```
# Solution: GELU Activation
Reference solution.
```python
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
def my_gelu(x):
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
```
```python
x = torch.tensor([-2., -1., 0., 1., 2.])
print('Output:', my_gelu(x))
print('Ref:   ', torch.nn.functional.gelu(x))
```
```python
```
# Solution: Kaiming Initialization
Reference solution.
```python
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
def kaiming_init(weight):
    fan_in = weight.shape[1] if weight.dim() >= 2 else weight.shape[0]
    std = math.sqrt(2.0 / fan_in)
    with torch.no_grad():
        weight.normal_(0, std)
    return weight
```
```python
import math
w = torch.empty(256, 512)
kaiming_init(w)
print(f'Mean: {w.mean():.4f} (expect ~0)')
print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')
```
```python
```
# Solution: Gradient Norm Clipping
Reference solution.
```python
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
def clip_grad_norm(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm.item()
```
```python
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Returned:', orig)
```
```python
```
# Solution: Gradient Accumulation
Reference solution.
```python
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
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)
```
```python
```
# 🟡 Solution: Linear Regression
Reference solution demonstrating closed-form, gradient descent, and nn.Linear approaches.
```python
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
class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation via augmented matrix."""
        N, D = X.shape
        X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, D+1)
        theta = torch.linalg.lstsq(X_aug, y).solution      # (D+1,)
        w = theta[:D]
        b = theta[D]
        return w.detach(), b.detach()
    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient computation — no autograd."""
        N, D = X.shape
        w = torch.zeros(D)
        b = torch.tensor(0.0)
        for _ in range(steps):
            pred = X @ w + b          # (N,)
            error = pred - y           # (N,)
            grad_w = (2.0 / N) * (X.T @ error)  # (D,)
            grad_b = (2.0 / N) * error.sum()     # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b
        return w, b
    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """PyTorch nn.Linear with autograd training loop."""
        N, D = X.shape
        layer = nn.Linear(D, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(steps):
            optimizer.zero_grad()
            pred = layer(X).squeeze(-1)  # (N,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()
        w = layer.weight.data.squeeze(0)  # (D,)
        b = layer.bias.data.squeeze(0)    # scalar ()
        return w, b
```
```python
torch.manual_seed(42)
X = torch.randn(100, 3)
true_w = torch.tensor([2.0, -1.0, 0.5])
y = X @ true_w + 3.0
model = LinearRegression()
for name, method in [("Closed-form", model.closed_form),
                      ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                      ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
    w, b = method(X, y)
    print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")
```
```python
```
# 🟡 Solution: Simple Linear Layer
Reference solution for a fully-connected linear layer: **y = xW^T + b**
```python
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
class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
```
```python
layer = SimpleLinear(8, 4)
print("W shape:", layer.weight.shape)
print("b shape:", layer.bias.shape)
x = torch.randn(2, 8)
print("Output shape:", layer.forward(x).shape)
```
```python
```
# 🟡 Solution: Implement LayerNorm
Reference solution for Layer Normalization.
$$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$
```python
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
def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta
```
```python
x = torch.randn(2, 8)
gamma = torch.ones(8)
beta = torch.zeros(8)
out = my_layer_norm(x, gamma, beta)
ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
print("Match ref?", torch.allclose(out, ref, atol=1e-4))
```
```python
```
# 🟡 Solution: Implement BatchNorm
Reference solution for Batch Normalization with both **training** and **inference** behavior, including running mean/variance updates.
```python
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
x = torch.randn(8, 4)
gamma = torch.ones(4)
beta = torch.zeros(4)
running_mean = torch.zeros(4)
running_var = torch.ones(4)
out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
print("[Train] Column means:", out_train.mean(dim=0))
print("[Train] Column stds: ", out_train.std(dim=0))
print("Updated running_mean:", running_mean)
print("Updated running_var:", running_var)
out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
print("[Eval] Column means (using running stats):", out_eval.mean(dim=0))
```
```python
```
# 🟡 Solution: Implement RMSNorm
Reference solution for Root Mean Square Normalization.
```python
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
def rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x / rms * weight
```
```python
x = torch.randn(2, 8)
out = rms_norm(x, torch.ones(8))
print('RMS of output:', out.pow(2).mean(dim=-1).sqrt())
```
```python
```
# 🟠 Solution: SwiGLU MLP
Reference solution — gated feed-forward network used in LLaMA, Mistral, and PaLM.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```
```python
class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff)
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```
```python
mlp = SwiGLUMLP(d_model=64, d_ff=128)
x = torch.randn(2, 8, 64)
print('Output:', mlp(x).shape)
print('Params:', sum(p.numel() for p in mlp.parameters()))
```
```python
```
# Solution: 2D Convolution
Reference solution.
```python
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
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print('Output:', my_conv2d(x, w).shape)
print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
```
```python
```
# Solution: Multi-Head Cross-Attention
Reference solution.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x_q, x_kv):
        B, S_q, _ = x_q.shape
        S_kv = x_kv.shape[1]
        q = self.W_q(x_q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x_kv).view(B, S_kv, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S_q, -1))
```
```python
attn = MultiHeadCrossAttention(64, 4)
x_q = torch.randn(2, 6, 64)
x_kv = torch.randn(2, 10, 64)
print('Output:', attn(x_q, x_kv).shape)
```
```python
```
# 🔴 Solution: Softmax Attention
Reference solution for the core Transformer attention mechanism.
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
```python
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
def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```
```python
torch.manual_seed(42)
Q = torch.randn(2, 4, 8)
K = torch.randn(2, 4, 8)
V = torch.randn(2, 4, 8)
out = scaled_dot_product_attention(Q, K, V)
print("Output shape:", out.shape)
print("Attention weights sum to 1?", True)
Q2 = torch.randn(1, 3, 16)
K2 = torch.randn(1, 5, 16)
V2 = torch.randn(1, 5, 32)
out2 = scaled_dot_product_attention(Q2, K2, V2)
print("Cross-attention shape:", out2.shape, "(expected: 1, 3, 32)")
```
```python
```
# 🔴 Solution: Multi-Head Attention
Reference solution for the Multi-Head Attention mechanism.
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, Q, K, V):
        B, S_q, _ = Q.shape
        S_k = K.shape[1]
        q = self.W_q(Q).view(B, S_q, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(B, S_k, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S_q, -1)
        return self.W_o(out)
```
```python
torch.manual_seed(0)
mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha.forward(x, x, x)
print("Self-attn shape:", out.shape)
Q = torch.randn(1, 3, 32)
K = torch.randn(1, 7, 32)
V = torch.randn(1, 7, 32)
out2 = mha.forward(Q, K, V)
print("Cross-attn shape:", out2.shape)
```
```python
```
# 🔴 Solution: Causal Self-Attention
Reference solution — softmax attention with an upper-triangular mask.
```python
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
def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```
```python
torch.manual_seed(0)
Q = torch.randn(1, 4, 8)
K = torch.randn(1, 4, 8)
V = torch.randn(1, 4, 8)
out = causal_attention(Q, K, V)
print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))
```
```python
```
# 🔴 Solution: Grouped Query Attention
Reference solution for GQA — MHA with shared KV heads.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_kv_heads, self.d_k).transpose(1, 2)
        repeats = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = attn.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out)
```
```python
gqa = GroupQueryAttention(32, 8, 2)
print('Output:', gqa.forward(torch.randn(1, 4, 32)).shape)
```
```python
```
# 🔴 Solution: Sliding Window Attention
Reference solution — softmax attention with a band mask.
```python
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
def sliding_window_attention(Q, K, V, window_size):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)
```
```python
Q=torch.randn(1,6,8); K=torch.randn(1,6,8); V=torch.randn(1,6,8)
print('window=0==V?', torch.allclose(sliding_window_attention(Q,K,V,0), V, atol=1e-5))
```
```python
```
# 🔴 Solution: Linear Self-Attention
Reference solution — kernel-based attention with elu+1 feature map.
```python
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
def linear_attention(Q, K, V):
    Q_prime = F.elu(Q) + 1
    K_prime = F.elu(K) + 1
    KV = torch.bmm(K_prime.transpose(1, 2), V)       # (B, D_k, D_v)
    Z = K_prime.sum(dim=1, keepdim=True)              # (B, 1, D_k)
    num = torch.bmm(Q_prime, KV)                      # (B, S, D_v)
    den = torch.bmm(Q_prime, Z.transpose(1, 2))       # (B, S, 1)
    return num / (den + 1e-6)
```
```python
Q=torch.randn(1,8,16); K=torch.randn(1,8,16); V=torch.randn(1,8,32)
print('Shape:', linear_attention(Q,K,V).shape)
```
```python
```
# 🔴 Solution: KV Cache Attention
Reference solution — multi-head attention with KV caching for autoregressive inference.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    def forward(self, x, cache=None):
        B, S_new, _ = x.shape
        q = self.W_q(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S_new, self.num_heads, self.d_k).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        S_total = k.shape[2]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if S_new > 1:
            S_past = S_total - S_new
            mask = torch.triu(
                torch.ones(S_new, S_total, device=x.device, dtype=torch.bool),
                diagonal=S_past + 1,
            )
            scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        out = self.W_o(attn.transpose(1, 2).contiguous().view(B, S_new, -1))
        return out, new_cache
```
```python
torch.manual_seed(0)
attn = KVCacheAttention(d_model=64, num_heads=4)
x = torch.randn(1, 6, 64)
full_out, _ = attn(x)
out1, cache = attn(x[:, :4])
out2, cache = attn(x[:, 4:5], cache=cache)
out3, cache = attn(x[:, 5:6], cache=cache)
inc_out = torch.cat([out1, out2, out3], dim=1)
print('Full shape:', full_out.shape)
print('Match:', torch.allclose(full_out, inc_out, atol=1e-5))
print('Final cache K shape:', cache[0].shape)
```
```python
```
# Solution: Rotary Position Embedding (RoPE)
Reference solution.
```python
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
def apply_rope(q, k):
    B, S, D = q.shape
    pos = torch.arange(S, device=q.device).unsqueeze(1).float()
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (10000.0 ** (dim / D))
    angles = pos * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)
    def rotate(x):
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x1 * cos_a - x2 * sin_a,
                            x1 * sin_a + x2 * cos_a], dim=-1).flatten(-2)
    return rotate(q), rotate(k)
```
```python
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))
```
```python
```
# Solution: Flash Attention (Tiled)
Reference solution.
```python
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
def flash_attention(Q, K, V, block_size=32):
    B, S, D = Q.shape
    output = torch.zeros_like(Q)
    for i in range(0, S, block_size):
        qi = Q[:, i:i+block_size]
        bs_q = qi.shape[1]
        row_max = torch.full((B, bs_q, 1), float('-inf'), device=Q.device)
        row_sum = torch.zeros(B, bs_q, 1, device=Q.device)
        acc = torch.zeros(B, bs_q, D, device=Q.device)
        for j in range(0, S, block_size):
            kj = K[:, j:j+block_size]
            vj = V[:, j:j+block_size]
            scores = torch.bmm(qi, kj.transpose(1, 2)) / math.sqrt(D)
            block_max = scores.max(dim=-1, keepdim=True).values
            new_max = torch.maximum(row_max, block_max)
            correction = torch.exp(row_max - new_max)
            exp_scores = torch.exp(scores - new_max)
            acc = acc * correction + torch.bmm(exp_scores, vj)
            row_sum = row_sum * correction + exp_scores.sum(dim=-1, keepdim=True)
            row_max = new_max
        output[:, i:i+block_size] = acc / row_sum
    return output
```
```python
import math
Q, K, V = torch.randn(1, 16, 8), torch.randn(1, 16, 8), torch.randn(1, 16, 8)
out = flash_attention(Q, K, V, block_size=4)
scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(8)
ref = torch.bmm(torch.softmax(scores, dim=-1), V)
print('Shape:', out.shape)
print('Max diff:', (out - ref).abs().max().item())
```
```python
```
# Solution: LoRA (Low-Rank Adaptation)
Reference solution.
```python
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
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad_(False)
        self.linear.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank
    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
```
```python
layer = LoRALinear(16, 8, rank=4)
x = torch.randn(2, 16)
print('Output:', layer(x).shape)
trainable = sum(p.numel() for p in layer.parameters() if p.requires_grad)
total = sum(p.numel() for p in layer.parameters())
print(f'Trainable: {trainable}/{total} ({100*trainable/total:.1f}%)')
```
```python
```
# Solution: ViT Patch Embedding
Reference solution.
```python
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
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size
        n_h, n_w = H // p, W // p
        x = x.reshape(B, C, n_h, p, n_w, p)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, n_h * n_w, C * p * p)
        return self.proj(x)
```
```python
pe = PatchEmbedding(224, 16, 3, 768)
x = torch.randn(1, 3, 224, 224)
print('Output:', pe(x).shape)
print('Patches:', pe.num_patches)
```
```python
```
# 🔴 Solution: GPT-2 Transformer Block
Reference solution — pre-norm, causal self-attention, 4x MLP with GELU.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import torch
import torch.nn as nn
import math
```
```python
class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
    def _attn(self, x):
        B, S, _ = x.shape
        q = self.W_q(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float('-inf'))
        weights = torch.softmax(scores, dim=-1)
        attn = torch.matmul(weights, v)
        return self.W_o(attn.transpose(1, 2).contiguous().view(B, S, -1))
    def forward(self, x):
        x = x + self._attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
```
```python
block = GPT2Block(64, 4)
print('Output:', block(torch.randn(2, 8, 64)).shape)
print('Params:', sum(p.numel() for p in block.parameters()))
```
```python
```
# Solution: Mixture of Experts (MoE)
Reference solution.
```python
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
class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
            for _ in range(num_experts)
        ])
    def forward(self, x):
        orig_shape = x.shape
        if x.dim() == 3:
            B, S, D = x.shape
            x_flat = x.reshape(-1, D)
        else:
            x_flat = x
        logits = self.router(x_flat)
        top_vals, top_idx = logits.topk(self.top_k, dim=-1)
        weights = torch.softmax(top_vals, dim=-1)
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            for e in range(len(self.experts)):
                mask = (top_idx[:, k] == e)
                if mask.any():
                    output[mask] += weights[mask, k:k+1] * self.experts[e](x_flat[mask])
        return output.reshape(orig_shape)
```
```python
moe = MixtureOfExperts(32, 64, num_experts=4, top_k=2)
x = torch.randn(2, 8, 32)
print('Output:', moe(x).shape)
print('Params:', sum(p.numel() for p in moe.parameters()))
```
```python
```
# Solution: Adam Optimizer
Reference solution.
```python
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
class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
    def step(self):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
                self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                p -= self.lr * m_hat / (torch.sqrt(v_hat) + self.eps)
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```
```python
torch.manual_seed(0)
w = torch.randn(4, 3, requires_grad=True)
opt = MyAdam([w], lr=0.01)
for i in range(5):
    loss = (w ** 2).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    print(f'Step {i}: loss={loss.item():.4f}')
```
```python
```
# Solution: Cosine LR Scheduler with Warmup
Reference solution.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
import math
```
```python
def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
```
```python
lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
print(f'Start: {lrs[0]:.6f}, Warmup end: {lrs[10]:.6f}, Mid: {lrs[55]:.6f}, End: {lrs[100]:.6f}')
```
```python
```
# Solution: Top-k / Top-p Sampling
Reference solution.
```python
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
def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_k_val = logits.topk(top_k).values[-1]
        logits[logits < top_k_val] = float('-inf')
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum - probs) > top_p
        sorted_logits[mask] = float('-inf')
        logits = torch.empty_like(logits).scatter_(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).item()
```
```python
logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))
```
```python
```
# Solution: Beam Search Decoding
Reference solution.
```python
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
def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]
    completed = []
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue
            log_probs = log_prob_fn(torch.tensor(seq))
            topk_lp, topk_idx = log_probs.topk(beam_width)
            for j in range(beam_width):
                candidates.append((score + topk_lp[j].item(), seq + [topk_idx[j].item()]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    all_seqs = completed + beams
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    return all_seqs[0][1]
```
```python
def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp
seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
print('Sequence:', seq)
```
```python
```
# Solution: Speculative Decoding
Reference solution.
```python
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
torch.manual_seed(0)
probs = torch.softmax(torch.randn(4, 10), dim=-1)
tokens = torch.tensor([2, 5, 1, 8])
print('Perfect draft:', speculative_decode(probs, probs, tokens))
```
```python
```
# Solution: Byte-Pair Encoding (BPE)
Reference solution.
```python
try:
    import google.colab
    get_ipython().run_line_magic('pip', 'install -q torch-judge')
except ImportError:
    pass
```
```python
```
```python
class SimpleBPE:
    def __init__(self):
        self.merges = []
    def train(self, corpus, num_merges):
        vocab = {}
        for word in corpus:
            symbols = tuple(word) + ('</w>',)
            vocab[symbols] = vocab.get(symbols, 0) + 1
        self.merges = []
        for _ in range(num_merges):
            pairs = {}
            for word, freq in vocab.items():
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pairs[pair] = pairs.get(pair, 0) + freq
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_vocab = {}
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                        new_word.append(word[i] + word[i + 1])
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            vocab = new_vocab
    def encode(self, text):
        all_tokens = []
        for word in text.split():
            symbols = list(word) + ['</w>']
            for a, b in self.merges:
                i = 0
                while i < len(symbols) - 1:
                    if symbols[i] == a and symbols[i + 1] == b:
                        symbols = symbols[:i] + [a + b] + symbols[i + 2:]
                    else:
                        i += 1
            all_tokens.extend(symbols)
        return all_tokens
```
```python
bpe = SimpleBPE()
bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
print('Merges:', bpe.merges)
print('Encode:', bpe.encode('low lower newest'))
```
```python
```
# Solution: INT8 Quantized Linear
Reference solution.
```python
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
class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        scale = weight.abs().amax(dim=1, keepdim=True) / 127.0
        self.register_buffer('weight_int8',
            torch.round(weight / (scale + 1e-10)).clamp(-128, 127).to(torch.int8))
        self.register_buffer('scale', scale)
        self.bias = nn.Parameter(bias.clone()) if bias is not None else None
    def forward(self, x):
        w = self.weight_int8.float() * self.scale
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out
```
```python
w = torch.randn(8, 4)
q = Int8Linear(w)
print('Output:', q(torch.randn(2, 4)).shape)
print('Weight dtype:', q.weight_int8.dtype)
print('Compression: float32 -> int8 = 4x')
```
```python
```
# Solution: DPO (Direct Preference Optimization) Loss
Reference solution.
```python
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
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```
```python
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
```
```python
```
# GRPO

GRPO 的核心是：**对同一个 prompt 采样多个回答，用组内 reward 的均值和方差构造 advantage，代替 PPO 里的 critic。**
训练目标仍然沿用 **PPO 的 clipped objective**，再加一个 **reference model 的 KL 约束**，防止策略漂移过大。DeepSeekMath 把 GRPO描述为 PPO 的一种变体，用组内相对分数替代 value function；TRL（Transformer Reinforcement Learning）文档也沿用这一思路，并给出带 clipped surrogate objective 的实现表述。 ([arXiv][1])

先看整体。GRPO 可以概括成三步。

第一步，按组计算 advantage：

$$
A_i=\frac{r_i-\mu_g}{\sigma_g+\epsilon},
\qquad
\mu_g=\frac{1}{|g|}\sum_{j\in g}r_j,
\qquad
\sigma_g=\sqrt{\frac{1}{|g|}\sum_{j\in g}(r_j-\mu_g)^2}
$$

第二步，构造 PPO 的 ratio 和 clipped objective：

$$
\rho_i=\exp(\log \pi_{\text{new},i}-\log \pi_{\text{old},i})
$$

$$
L_{\text{policy}}
=
-\frac{1}{B}\sum_{i=1}^{B}
\min\Big(
\rho_i A_i,;
\mathrm{clip}(\rho_i,1-\epsilon_{\text{clip}},1+\epsilon_{\text{clip}})A_i
\Big)
$$

第三步，加上 reference model 的约束：

$$
L_{\text{KL}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left(\log \pi_{\text{new},i}-\log \pi_{\text{ref},i}\right)
$$

最终 loss：

$$
L=L_{\text{policy}}+\beta L_{\text{KL}}
$$

上面这组公式抓住了 DeepSeekMath 风格 GRPO 的主线：组内相对优势、PPO 式裁剪、参考模型约束。TRL 文档中也给出等价思路，并说明在一些设置下 clipping 的作用会被弱化或退化到原始目标。 ([arXiv][1])

##### 代码骨架

```py
import torch

def grpo_loss(old_logps, new_logps, ref_logps, rewards, group_ids,
              clip_eps=0.2, beta=0.01, eps=1e-8):
    adv = torch.empty_like(rewards)
```

##### （1）到（3）组内均值、标准差、标准化 advantage

$$
\mu_g=\frac{1}{|g|}\sum_{j\in g}r_j
$$

$$
\sigma_g=\sqrt{\frac{1}{|g|}\sum_{j\in g}(r_j-\mu_g)^2}
$$

$$
A_i=\frac{r_i-\mu_g}{\sigma_g+\epsilon}
$$

```py
    for gid in group_ids.unique():
        mask = (group_ids == gid)
        r = rewards[mask]
        mu_g = r.mean()
        sigma_g = r.std(unbiased=False)
        adv[mask] = (r - mu_g) / (sigma_g + eps)
```

##### （4）advantage 不反传梯度

这里把 advantage 当成常数使用：

$$
A_i \text{ is treated as a constant}
$$

```py
    adv = adv.detach()
```

##### （5）策略比值 ratio

$$
\rho_i=\frac{\pi_{\text{new}}(y_i|x_i)}{\pi_{\text{old}}(y_i|x_i)}
=\exp(\log \pi_{\text{new},i}-\log \pi_{\text{old},i})
$$

```py
    ratio = torch.exp(new_logps - old_logps)
```

##### （6）未裁剪目标

$$
L_1^{(i)}=\rho_i A_i
$$

```py
    obj1 = ratio * adv
```

##### （7）裁剪后的目标

$$
L_2^{(i)}=
\mathrm{clip}(\rho_i,1-\epsilon_{\text{clip}},1+\epsilon_{\text{clip}}),A_i
$$

```py
    obj2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
```

##### （8）PPO policy loss

$$
L_{\text{policy}}
=
-\frac{1}{B}\sum_{i=1}^{B}\min(L_1^{(i)},L_2^{(i)})
$$

```py
    policy_loss = -torch.min(obj1, obj2).mean()
```

##### （9）reference model 的 KL 约束

这里用一个常见的简化形式：

$$
L_{\text{KL}}
=
\frac{1}{B}\sum_{i=1}^{B}
\left(\log \pi_{\text{new},i}-\log \pi_{\text{ref},i}\right)
$$

```py
    kl_loss = (new_logps - ref_logps).mean()
```

##### （10）最终 GRPO loss

$$
L=L_{\text{policy}}+\beta L_{\text{KL}}
$$

```py
    loss = policy_loss + beta * kl_loss
    return loss
```

```py
import torch

def grpo_loss(old_logps, new_logps, ref_logps, rewards, group_ids,
              clip_eps=0.2, beta=0.01, eps=1e-8):
    adv = torch.empty_like(rewards)

    for gid in group_ids.unique():
        mask = (group_ids == gid)
        r = rewards[mask]
        mu_g = r.mean()
        sigma_g = r.std(unbiased=False)
        adv[mask] = (r - mu_g) / (sigma_g + eps)

    adv = adv.detach()
    ratio = torch.exp(new_logps - old_logps)

    obj1 = ratio * adv
    obj2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    policy_loss = -torch.min(obj1, obj2).mean()

    kl_loss = (new_logps - ref_logps).mean()

    loss = policy_loss + beta * kl_loss
    return loss
```

```py
old_logps = torch.tensor([-0.2, -0.6, -1.1, -1.4])
new_logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
ref_logps = torch.tensor([-0.1, -0.55, -1.05, -1.45])

rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])

print("Loss:", grpo_loss(old_logps, new_logps, ref_logps, rewards, group_ids).item())
```

##### 设计理念和常见追问

**1. 为什么要有 reference model？它和外面的 KL 项是什么关系？**
reference model 是“被拿来比较的锚点策略”，KL 项是“度量当前策略偏离这个锚点有多远的惩罚”。两者不是一回事：reference model 是参照对象，KL 是约束形式。没有 reference model，KL 项就没有明确的比较基准；只有 reference model 但没有 KL 惩罚，训练也可能迅速偏离初始分布，导致模式坍塌、语言质量下降或奖励黑化。DeepSeekMath 明确引入了 KL 系数；TRL 的日志里也把 `objective/kl` 当成需要监控的重要量。 ([arXiv][1])

**2. reference model 和 old policy 能不能互换？**
通常不能。old policy 的职责是做 importance ratio，也就是告诉你“当前更新相对采样策略偏了多少”；reference model 的职责是做分布锚定，限制策略不要偏离某个稳定起点。前者是优化几何里的“更新基线”，后者是对齐里的“行为基准”。把它们合并，等于把“采样分布约束”和“参考分布约束”混成一件事，会让训练目标变味。PPO 原论文里 old policy 是 ratio 的分母；DeepSeekMath/TRL 里的 reference policy 则服务于 KL 约束，这两层角色是分开的。 ([arXiv][2])

**3. KL 方向能不能反过来？比如从 (D_{\mathrm{KL}}(\pi_{\text{new}}|\pi_{\text{ref}})) 换成 (D_{\mathrm{KL}}(\pi_{\text{ref}}|\pi_{\text{new}}))**
一般不随便换。KL 不是对称的，两个方向惩罚的行为不同。
(D_{\mathrm{KL}}(\pi_{\text{new}}|\pi_{\text{ref}})) 更强调“当前策略不要把概率压到 reference 很小的地方”，常见于生成模型和 RLHF / RLVR（Reinforcement Learning from Human/Verifier Rewards）里，因为它直接约束当前策略的偏移。
反方向 (D_{\mathrm{KL}}(\pi_{\text{ref}}|\pi_{\text{new}})) 会更强调覆盖 reference 的高概率区域，优化特性和数值行为都不同。PPO 原论文就区分了不同 KL 罚项；在语言模型对齐中，工程实现通常不会把两个方向随意互换。 ([arXiv][3])

**4. 代码里为什么常写成 `new_logps - ref_logps`，这真的是 KL 吗？**
严格说，这往往是一个**采样路径上的 KL 近似项**，不是完整分布积分形式的精确 KL。工程里这样做是因为你只拿到了采样动作对应的 logprob，计算便宜，方向也对。TRL 文档把这类项作为实际训练中的 KL 估计；DeepSeekMath 也是在训练目标里显式使用 KL 正则，而不是要求每次都精确积分整分布。面试时说“这是 token / sample-level 的 KL proxy（代理项）”最稳。 ([Hugging Face][4])

**5. clip 为什么要放在 ratio 上？能不能不 clip？**
clip 的作用是限制单次更新幅度，让多轮小步更新更稳定。这正是 PPO 的核心设计之一。原始 PPO 论文把 clipped surrogate objective 作为主角，和 KL-penalty 版本做过比较；后来还有不少工作研究“是否一定需要 ratio clipping”或改成别的约束方式，说明它不是唯一可行手段，但它之所以流行，就是因为简单且稳。GRPO 继承了这套思想。 ([arXiv][2])

**6. `min(obj1, obj2)` 能不能换成 `max(obj1, obj2)`？**
不能。这里的 `min` 是 PPO clipped surrogate 的关键。它给的是一个**保守下界**式目标：当 ratio 朝着“有利于把优势样本继续放大、或把劣势样本继续压低”的方向跑太远时，clipping 会截住它，不让目标继续无约束增长。
如果改成 `max`，你就等于鼓励两边取更大的那个，恰好破坏“限制过大更新”的目的，训练会更激进，更容易失稳。PPO 的整个裁剪逻辑就是建立在这个 `min` 上。 ([arXiv][2])

**7. 外面这个负号能不能改成正号？**
如果你的代码框架是“最小化 loss”，外面必须是负号，因为我们真正想做的是**最大化**策略目标。把负号拿掉，就会变成最小化本来应该被最大化的东西，优化方向直接反了。这个问题本质上是“训练器在做 minimize 还是 maximize”，和 RL 目标本身的数学方向一致。PPO / GRPO 的论文表达通常是最大化 surrogate objective，而实际代码常写成带负号的 loss。 ([arXiv][2])

**8. clip 里的区间能不能只裁一边，或者两边都改？**
可以有变体。标准 PPO 是两边裁到 ([1-\epsilon, 1+\epsilon])。但后续工作和一些工程配方会改 clipping 细节，例如两侧裁剪、不同非对称裁剪，或者把裁剪对象从 ratio 改成别的量。Hugging Face 的 paper index 里就提到过 two-sided GRPO clipping 这样的变体；也有工作尝试不做 ratio clipping，而改做 KL clipping 或其他近端约束。换句话说，**clip 机制可以改，`为什么要限制更新步长` 这个原则不变。** ([Hugging Face][5])

**9. 既然有 KL，clip 还要不要？二者会不会重复？**
二者相关，但不完全重复。clip 主要约束“相对 old policy 的局部更新步长”，KL 更像“相对 reference policy 的全局分布锚定”。一个偏向优化稳定性，一个偏向对齐与分布保持。工程上有时会削弱其中一个，但把两者都保留通常更稳。TRL 文档里同时保留了 clipped objective 和 KL 监控；DeepSeekMath 也明确设置了 KL 系数。 ([arXiv][1])

**10. 有哪些方向在改进这些设计，尤其是 clip 和 KL？**
常见思路有三类。
第一类，**改 clip**：比如 two-sided clipping、截断 PPO、研究 clipping 理论性质，目标是让更新更稳或更高效。
第二类，**弱化或移除 KL/reference model**：有些近年的工作认为在某些 reasoning 场景，rule-based reward 足够强，reference model 可被弱化甚至移除，以节省显存和算力。
第三类，**重新解释 GRPO / group-relative 更新**：一些近期论文讨论 GRPO 的目标到底在优化什么、组相对优势与 off-policy / process reward 的关系。
这些都说明：GRPO 的核心框架很稳，但每个零件都还在被继续改。 ([arXiv][6])

**11. 面试里最稳的一句总括怎么说？**
可以直接说：
**GRPO 本质上是“去 critic 的 PPO for LLMs”：用组内相对 reward 构造 advantage，用 PPO clipping 控制更新幅度，再用 reference-model KL 约束策略不要漂太远。old policy 和 reference policy 角色不同，`min` 和负号都不能随便改，因为它们决定了保守更新和正确优化方向。** 这句话和 DeepSeekMath 的原始设定，以及 TRL 的工程实现逻辑是一致的。 ([arXiv][1])

[1]: https://arxiv.org/pdf/2402.03300?utm_source=chatgpt.com "DeepSeekMath"
[2]: https://arxiv.org/abs/1707.06347?utm_source=chatgpt.com "[1707.06347] Proximal Policy Optimization Algorithms"
[3]: https://arxiv.org/pdf/1707.06347?utm_source=chatgpt.com "Proximal Policy Optimization Algorithms"
[4]: https://huggingface.co/docs/trl/grpo_trainer?utm_source=chatgpt.com "GRPO Trainer"
[5]: https://huggingface.co/docs/trl/paper_index?utm_source=chatgpt.com "Paper Index"
[6]: https://arxiv.org/html/2602.03190v1?utm_source=chatgpt.com "Prompt Augmentation Scales up GRPO Training on ..."

# Solution: PPO Clipped Loss
Reference solution for the PPO clipped surrogate loss task.
```python
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
def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    """PPO clipped surrogate loss.
    new_logps: (B,) current policy log-probs
    old_logps: (B,) old policy log-probs (treated as constant)
    advantages: (B,) advantage estimates (treated as constant)
    returns: scalar loss (Tensor)
    """
    old_logps_detached = old_logps.detach()
    adv_detached = advantages.detach()
    ratios = torch.exp(new_logps - old_logps_detached)
    unclipped = ratios * adv_detached
    clipped = torch.clamp(ratios, 1.0 - clip_ratio, 1.0 + clip_ratio) * adv_detached
    return -torch.min(unclipped, clipped).mean()
```
```python
new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
print('Loss:', ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2))
```
```python
```
