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
