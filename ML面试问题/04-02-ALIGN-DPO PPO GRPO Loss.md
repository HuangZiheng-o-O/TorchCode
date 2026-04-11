##### GRPO

GRPO 的核心是：**对同一个 prompt 采样多个回答，用组内 reward 的均值和方差构造 advantage，代替 PPO 里的 critic。**  
训练目标仍然沿用 **PPO 的 clipped objective**，再加一个 **reference model 的 KL 约束**，防止策略漂移过大。

你可以直接按编号讲：

*   **（1）到（3）**：先在每个 group 内计算 reward 的均值和标准差，得到标准化 advantage。
*   **（4）到（5）**：再用当前策略和旧策略的 logprob 计算 ratio。
*   **（6）到（8）**：构造 PPO 的 clipped objective，防止更新过大。
*   **（9）**：再加一个相对 reference model 的 KL 约束。
*   **（10）**：两部分加起来就是最终的 GRPO loss。

##### （1）组内均值

对同一个 prompt 下的一组回答，先计算 reward 的均值：

$$
\mu_g = \frac{1}{|g|}\sum_{j \in g} r_j
$$

##### （2）组内标准差

$$
\sigma_g = \sqrt{\frac{1}{|g|}\sum_{j \in g}(r_j-\mu_g)^2}
$$

##### （3）组内标准化 advantage

$$
A_i = \frac{r_i - \mu_g}{\sigma_g + \epsilon}
$$

##### （4）策略比值

$$
\rho_i = \frac{\pi_{\text{new}}(y_i|x_i)}{\pi_{\text{old}}(y_i|x_i)}
$$

##### （5）logprob 形式的策略比值

$$
\rho_i = \exp(\log \pi_{\text{new},i} - \log \pi_{\text{old},i})
$$

##### （6）未裁剪目标

$$
L_1^{(i)} = \rho_i A_i
$$

##### （7）裁剪后的目标

$$
L_2^{(i)} = \mathrm{clip}(\rho_i, 1-\epsilon_{\text{clip}}, 1+\epsilon_{\text{clip}})\, A_i
$$

##### （8）PPO policy loss

$$
L_{\text{policy}} = -\frac{1}{B}\sum_{i=1}^{B}\min(L_1^{(i)}, L_2^{(i)})
$$

##### （9）参考模型 KL 约束

这里用一个常见的简化形式：

$$
L_{\text{KL}} = \frac{1}{B}\sum_{i=1}^{B}\left(\log \pi_{\text{new},i} - \log \pi_{\text{ref},i}\right)
$$

##### （10）最终 GRPO loss

$$
L = L_{\text{policy}} + \beta L_{\text{KL}}
$$



```py
import torch

def grpo_loss(old_logps, new_logps, ref_logps, rewards, group_ids,
              clip_eps=0.2, beta=0.01, eps=1e-8):
    # （1）（2）（3）按组计算均值、标准差、标准化 advantage:
    # mu_g = mean(r), sigma_g = std(r), A_i = (r_i - mu_g) / (sigma_g + eps)
    adv = torch.empty_like(rewards)
    for gid in group_ids.unique():
        mask = (group_ids == gid)
        r = rewards[mask]
        adv[mask] = (r - r.mean()) / (r.std(unbiased=False) + eps)

    # （4）（5）GRPO / PPO 中 advantage 视为常数，不对 advantage 反传梯度
    adv = adv.detach()

    # rho_i = pi_new / pi_old = exp(log pi_new - log pi_old)
    ratio = torch.exp(new_logps - old_logps)

    # （6）未裁剪目标: L1 = rho_i * A_i
    obj1 = ratio * adv

    # （7）裁剪目标: L2 = clip(rho_i, 1 - clip_eps, 1 + clip_eps) * A_i
    obj2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    # （8）PPO policy loss:
    # L_policy = -mean(min(L1, L2))
    policy_loss = -torch.min(obj1, obj2).mean()

    # （9）参考模型约束:
    # L_KL = mean(log pi_new - log pi_ref)
    kl_loss = (new_logps - ref_logps).mean()

    # （10）最终 GRPO loss:
    # L = L_policy + beta * L_KL
    loss = policy_loss + beta * kl_loss
    return loss
```

##### Demo

```py
old_logps = torch.tensor([-0.2, -0.6, -1.1, -1.4])
new_logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
ref_logps = torch.tensor([-0.1, -0.55, -1.05, -1.45])

rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])

print("Loss:", grpo_loss(old_logps, new_logps, ref_logps, rewards, group_ids).item())
```
