[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/duoan/TorchCode/blob/master/solutions/24_rope_solution.ipynb)
# Solution: Rotary Position Embedding (RoPE)
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
# Demo
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)
qr, kr = apply_rope(q, k)
print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))
```
```python
from torch_judge import check
check('rope')
```



# RoPE 旋转位置编码

RoPE（Rotary Position Embedding，旋转位置编码）的目标不是“把位置向量直接加到 token embedding 上”，而是对每个位置的 query/key 子向量做一个二维旋转（2D rotation），使 attention score 自然携带相对位置信息（relative position information）。它的数学核心对象就是：**把最后一维按偶奇配对后，对每对  $(x_{2i}, x_{2i+1})$  施加位置相关角度  $\theta_{p,i}$  的旋转矩阵**。你给的参考实现正是这一版最标准、最适合面试手写的写法；下面我严格按这套代码展开。
Global Equations
----------------

设输入

$$
q,k \in \mathbb{R}^{B\times S\times D}, \qquad D \text{ 为偶数}
$$

对第  $p$  个位置（position）、第  $i$  个二维通道对（2D pair），定义频率（frequency）

$$
\omega_i = 10000^{- \frac{2i}{D}}
$$

对应旋转角（angle）

$$
\theta_{p,i} = p \cdot \omega_i
$$

把最后一维的偶数位和奇数位配对：

$$
x^{(i)} = \begin{bmatrix} x_{2i}\\ x_{2i+1} \end{bmatrix}
$$

RoPE 对这一对做旋转：

$$
R(\theta_{p,i}) = \begin{bmatrix} \cos\theta_{p,i} & -\sin\theta_{p,i}\\ \sin\theta_{p,i} & \cos\theta_{p,i} \end{bmatrix}
$$

$$
\tilde{x}^{(i)} = R(\theta_{p,i}) x^{(i)}
$$

展开后就是

$$
\tilde{x}_{2i} = x_{2i}\cos\theta_{p,i} - x_{2i+1}\sin\theta_{p,i}
$$

$$
\tilde{x}_{2i+1} = x_{2i}\sin\theta_{p,i} + x_{2i+1}\cos\theta_{p,i}
$$

最终分别得到

$$
q_{\text{rope}} = \mathrm{RoPE}(q),\qquad k_{\text{rope}} = \mathrm{RoPE}(k)
$$

然后进入 attention：

$$
\mathrm{Attention}(Q,K,V) = \mathrm{softmax}\!\left(\frac{Q_{\text{rope}}K_{\text{rope}}^\top}{\sqrt{d_k}}\right)V
$$

这里最关键的不是“旋转了”，而是**同一频率对子在不同位置用不同角度旋转**，从而让点积里出现相对位置项。

Implementation Blocks
---------------------

### （A）PyTorch 版本（标准实现）

这版直接复用参考解法，结构与知识库中的 RoPE 解完全一致。

```python
import torch
from torch import Tensor

def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    """
    q, k: (B, S, D)
    return: q_rope, k_rope, both (B, S, D)
    """
    B, S, D = q.shape
    assert D % 2 == 0, "RoPE requires even hidden dimension"

    # pos: (S, 1)
    # 每个位置 p = 0, 1, ..., S-1
    pos = torch.arange(S, device=q.device).unsqueeze(1).float()

    # dim: (D/2,)
    # 只取偶数维索引 0, 2, 4, ...
    dim = torch.arange(0, D, 2, device=q.device).float()

    # freqs: (D/2,)
    # 第 i 对通道的旋转频率
    freqs = 1.0 / (10000.0 ** (dim / D))

    # angles: (S, D/2)
    # 广播后得到每个位置、每个二维对子对应的旋转角
    angles = pos * freqs

    # cos_a, sin_a: (S, D/2)
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def rotate(x: Tensor) -> Tensor:
        # x: (B, S, D)
        # x1: (B, S, D/2), 偶数位
        # x2: (B, S, D/2), 奇数位
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # 下面两项分别是旋转后的实部/虚部
        # 由于 cos_a / sin_a 是 (S, D/2)，会在 batch 维自动广播到 (B, S, D/2)
        y1 = x1 * cos_a - x2 * sin_a
        y2 = x1 * sin_a + x2 * cos_a

        # stack: (B, S, D/2, 2)
        # flatten(-2): (B, S, D)
        # 把每对旋转后的两个分量重新交错拼回最后一维
        return torch.stack([y1, y2], dim=-1).flatten(-2)

    return rotate(q), rotate(k)
```

```
# Demo
q = torch.randn(1, 8, 16)
k = torch.randn(1, 8, 16)

qr, kr = apply_rope(q, k)

print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))

# from torch_judge import check
# check('rope')
```

### （B）NumPy 版本（苛刻面试版）

这版与 PyTorch 版是**同一计算图**，只是把后端换成了 NumPy。

```python
import numpy as np

def apply_rope_numpy(q: np.ndarray, k: np.ndarray):
    """
    q, k: (B, S, D)
    return: q_rope, k_rope, both (B, S, D)
    """
    B, S, D = q.shape
    assert D % 2 == 0, "RoPE requires even hidden dimension"

    # pos: (S, 1)
    pos = np.arange(S, dtype=np.float32)[:, None]

    # dim: (D/2,)
    dim = np.arange(0, D, 2, dtype=np.float32)

    # freqs: (D/2,)
    freqs = 1.0 / (10000.0 ** (dim / D))

    # angles: (S, D/2)
    angles = pos * freqs

    # cos_a, sin_a: (S, D/2)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    def rotate(x: np.ndarray) -> np.ndarray:
        # x1, x2: (B, S, D/2)
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]

        # y1, y2: (B, S, D/2)
        y1 = x1 * cos_a - x2 * sin_a
        y2 = x1 * sin_a + x2 * cos_a

        # stack: (B, S, D/2, 2)
        y = np.stack([y1, y2], axis=-1)

        # reshape → (B, S, D)
        return y.reshape(B, S, D)

    return rotate(q), rotate(k)
```

```python
# Demo
q = np.random.randn(1, 8, 16).astype(np.float32)
k = np.random.randn(1, 8, 16).astype(np.float32)

qr, kr = apply_rope_numpy(q, k)

print('Shape preserved:', qr.shape == q.shape)
print('Norm preserved:', np.allclose(
    np.linalg.norm(q, axis=-1),
    np.linalg.norm(qr, axis=-1),
    atol=1e-4
))
```

Step-by-step Derivation
-----------------------

默认按 PyTorch 主实现展开。

```
import torch
from torch import Tensor

def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    pass
```

### Step 1：确定位置索引与二维通道对子

#### （i）公式

位置索引：

$$
p \in \{0,1,\dots,S-1\}
$$

二维对子索引：

$$
i \in \left\{0,1,\dots,\frac{D}{2}-1\right\}
$$

偶奇配对：

$$
(x_{2i}, x_{2i+1})
$$

#### （ii）代码

```
B, S, D = q.shape
assert D % 2 == 0, "RoPE requires even hidden dimension"

pos = torch.arange(S, device=q.device).unsqueeze(1).float()
dim = torch.arange(0, D, 2, device=q.device).float()
```

#### （iii）解释

`pos` 只描述序列位置，shape 是 `(S, 1)`。  
`dim` 只取偶数维索引 `0, 2, 4, ...`，因为 RoPE 的基本单元不是单个标量，而是最后一维上的一个二维对子 `(2i, 2i+1)`。

### Step 2：定义每个二维对子对应的频率

#### （i）公式

$$
\omega_i = 10000^{- \frac{2i}{D}}
$$

由于代码里的 `dim = 0, 2, 4, ...`，所以直接写成

$$
\omega = 10000^{- \frac{\text{dim}}{D}}
$$

#### （ii）代码

```
freqs = 1.0 / (10000.0 ** (dim / D))
```

#### （iii）解释

低维对子旋转更快，高维对子旋转更慢。这和 Transformer 正弦位置编码（sinusoidal positional encoding）的频率设计是一致的。这里不是把 sin/cos 直接加到表示里，而是把它们变成旋转角速度（angular frequency）。

### Step 3：把“位置”映射成“旋转角”

#### （i）公式

$$
\theta_{p,i} = p \cdot \omega_i
$$

#### （ii）代码

```
angles = pos * freqs
cos_a = torch.cos(angles)
sin_a = torch.sin(angles)
```

#### （iii）解释

这里 `pos: (S, 1)`，`freqs: (D/2,)`，广播（broadcasting）后：

*   `angles`: `(S, D/2)`
*   `cos_a`: `(S, D/2)`
*   `sin_a`: `(S, D/2)`

意思是：**每个位置、每个二维对子，都有一个单独的旋转角**。

### Step 4：把最后一维拆成偶数位和奇数位

#### （i）公式

对任意输入  $x \in \mathbb{R}^{B\times S\times D}$ ，定义

$$
x_1 = x[..., 0::2], \qquad x_2 = x[..., 1::2]
$$

即

$$
x_1 \in \mathbb{R}^{B\times S\times D/2}, \qquad x_2 \in \mathbb{R}^{B\times S\times D/2}
$$

#### （ii）代码

```
def rotate(x: Tensor) -> Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
```

#### （iii）解释

RoPE 不是对整个向量一次性乘一个  $D\times D$  大矩阵；它实际上是对最后一维中的每个二维块 independently 做同一个形式的 2D 旋转。

### Step 5：对每个二维对子做旋转

#### （i）公式

$$
\begin{bmatrix} \tilde{x}_{2i}\\ \tilde{x}_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos\theta_{p,i} & -\sin\theta_{p,i}\\ \sin\theta_{p,i} & \cos\theta_{p,i} \end{bmatrix} \begin{bmatrix} x_{2i}\\ x_{2i+1} \end{bmatrix}
$$

展开得到

$$
\tilde{x}_{2i} = x_{2i}\cos\theta_{p,i} - x_{2i+1}\sin\theta_{p,i}
$$

$$
\tilde{x}_{2i+1} = x_{2i}\sin\theta_{p,i} + x_{2i+1}\cos\theta_{p,i}
$$

#### （ii）代码

```
    y1 = x1 * cos_a - x2 * sin_a
    y2 = x1 * sin_a + x2 * cos_a
```

#### （iii）解释

这是标准二维旋转矩阵。  
`x1, x2` 是 `(B, S, D/2)`，`cos_a, sin_a` 是 `(S, D/2)`，会沿 batch 维自动广播。也就是说：同一个 batch 内，相同位置和相同通道对子使用同一旋转角。

### Step 6：把旋转后的二维对子拼回原始维度

#### （i）公式

若每个对子旋转后得到

$$
(\tilde{x}_{2i}, \tilde{x}_{2i+1})
$$

则最终输出仍为

$$
\tilde{x} \in \mathbb{R}^{B\times S\times D}
$$

#### （ii）代码

```
    return torch.stack([y1, y2], dim=-1).flatten(-2)
```

#### （iii）解释

这里非常关键：

*   `y1, y2`: `(B, S, D/2)`
*   `stack(..., dim=-1)`: `(B, S, D/2, 2)`
*   `flatten(-2)`: `(B, S, D)`

这一步把每对旋转结果恢复成交错排列：

$$
[\tilde{x}_0,\tilde{x}_1,\tilde{x}_2,\tilde{x}_3,\dots]
$$

### Step 7：同样作用到  $q$  和  $k$ 

#### （i）公式

$$
q_{\text{rope}} = \mathrm{RoPE}(q), \qquad k_{\text{rope}} = \mathrm{RoPE}(k)
$$

#### （ii）代码

```
return rotate(q), rotate(k)
```

#### （iii）解释

RoPE 通常只作用在 query/key，不作用在 value。因为 attention score 来自  $QK^\top$ ，相对位置信息应该体现在匹配关系里，而不是直接改变 value 聚合内容本身。

### Step 8：完整函数

```
import torch
from torch import Tensor

def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
    B, S, D = q.shape
    assert D % 2 == 0, "RoPE requires even hidden dimension"

    pos = torch.arange(S, device=q.device).unsqueeze(1).float()
    dim = torch.arange(0, D, 2, device=q.device).float()
    freqs = 1.0 / (10000.0 ** (dim / D))
    angles = pos * freqs
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    def rotate(x: Tensor) -> Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack(
            [
                x1 * cos_a - x2 * sin_a,
                x1 * sin_a + x2 * cos_a,
            ],
            dim=-1
        ).flatten(-2)

    return rotate(q), rotate(k)
```

Design Insights / 面试追问
----------------------

### 1）为什么一定是“偶奇配对 + 2D rotation”？

因为 RoPE 的核心不是给每一维单独乘一个标量，而是构造\*\*保长度（norm-preserving）\*\*的二维旋转：

$$
\tilde{x}_{2i}^2 + \tilde{x}_{2i+1}^2 = x_{2i}^2 + x_{2i+1}^2
$$

所以它不会像简单加位置偏置那样直接改变向量模长（norm），而是改变方向（direction）。你 demo 里检查 `Norm preserved`，本质就是在验证这一点。

### 2）为什么 RoPE 能编码“相对位置”？

因为对于位置  $p$  和  $p'$ ，旋转后的点积可以化成与角差

$$
\theta_{p,i} - \theta_{p',i}
$$

有关，而不是只和绝对位置分别有关。也就是：

$$
R(\theta_p)^\top R(\theta_{p'}) = R(\theta_{p'} - \theta_p)
$$

所以 attention score 中自然出现相对位移  $(p'-p)$  的信息。这是它相比绝对位置加法编码（absolute additive positional encoding）最重要的结构优势。

### 3）为什么只对  $Q, K$  做，不对  $V$  做？

attention 权重来自

$$
QK^\top
$$

位置信息应该进入“谁关注谁”的相似度计算，而不是直接污染被聚合的内容值（value content）。  
如果把  $V$  也旋转，等于把内容表示本身强行位置化，通常没有必要，也不符合主流实现。

### 4）为什么频率是  $10000^{-2i/D}$  这一类形式？

这是多尺度（multi-scale）设计。低维对子对应高频，适合编码局部细粒度位置变化；高维对子对应低频，适合编码更长程的位置变化。  
如果所有对子都用同一个频率，那么模型只能感受到单一尺度的位置变化。

### 5）为什么代码里 `dim / D` 就够了？

因为 `dim` 本身取的是 `0, 2, 4, ...`：

```
dim = torch.arange(0, D, 2)
```

所以当 `dim = 2i` 时：

$$
\frac{\text{dim}}{D} = \frac{2i}{D}
$$

正好就是理论式里的指数。

### 6）为什么 `stack(...).flatten(-2)` 能恢复原顺序？

因为先得到

$$
[y1_i, y2_i]
$$

再按最后两维展平，顺序就是

$$
[y1_0, y2_0, y1_1, y2_1, \dots]
$$

也就是

$$
[\tilde{x}_0,\tilde{x}_1,\tilde{x}_2,\tilde{x}_3,\dots]
$$

这一步如果写错，比如直接 `cat([y1, y2], dim=-1)`，就会变成

$$
[\tilde{x}_0,\tilde{x}_2,\tilde{x}_4,\dots,\tilde{x}_1,\tilde{x}_3,\tilde{x}_5,\dots]
$$

语义就坏了。

### 7）为什么要求  $D$  是偶数？

因为最后一维必须拆成若干个二维对子。如果  $D$  是奇数，最后会剩一个维度无法配对，也就无法套用旋转矩阵。

### 8）面试里最容易被追问的一个点

不是“背代码”，而是这句：

$$
\tilde{x}_{2i} = x_{2i}\cos\theta - x_{2i+1}\sin\theta,\qquad \tilde{x}_{2i+1} = x_{2i}\sin\theta + x_{2i+1}\cos\theta
$$

你要能立刻说出：  
这就是二维旋转矩阵；RoPE 本质是在每个位置、每个频率对子上，对  $Q/K$  做位置相关旋转；因此  $QK^\top$  里自然携带相对位置信息，而不是额外加一个 position bias。

### 9）这份参考代码的优点

你给的实现有三个面试级优点：

1.  **最短计算图**：先造 `angles`，再 `cos/sin`，再 rotate。
2.  **广播自然**：`(S, D/2)` 自动广播到 `(B, S, D/2)`。
3.  **结构直接对应公式**：拆偶奇 → 旋转 → 拼回。

所以这就是最值得背的 RoPE 手写版。参考实现与知识库中的解一致。

如果你要，我下一条可以继续给你一版 **“RoPE + MultiHeadAttention 联合实现”**，把它直接嵌进 GPT/LLaMA 风格 attention 代码里。


