# SOE 训练流程详解:`L_IB` 的优化

> 本文件分两部分:
> - **第一部分**:精简工作流(训练对象、输入、前向、损失、梯度反传)
> - **第二部分**:工作流 → 代码的逐项对应(文件 + 行号)

---

# 第一部分:精简工作流

## 1. 训练对象:四个网络,分两组

| 组 | 网络 | 作用 |
|---|---|---|
| **base policy**(原版 Diffusion Policy) | **E** = 观测编码器(ResNet-18 + 可选 bottleneck) | 把观测压成观测嵌入 `c` |
| | **D** = 动作解码器(Diffusion UNet) | 把嵌入变成动作 chunk |
| **VIB 插件**(SOE 新增,16 维潜空间) | **p_θ** = 压缩器 | 把观测嵌入压成潜变量(高斯 `μ, σ`) |
| | **q_ϕ** = 还原器 | 把潜变量还原成嵌入 `c̃` |

架构示意:

```
              ┌────────── base policy(原版 Diffusion Policy)─────────┐
  观测 ──────→│  E:观测编码器  ──→  D:动作解码器  ──→  动作          │
              └──────────────────────────────────────────────────────┘
                       │ c                          ▲ D 为两条路共用
                       ▼
              ┌────────── VIB 插件(SOE 核心)──────────────────────┐
              │  p_θ(压缩) : c → 16维潜变量 (μ,σ)                │
              │  q_ϕ(还原) : 16维潜变量 → c̃                       │
              └──────────────────────────────────────────────────────┘
```

## 2. 输入与形状

每个训练 batch 输入两样东西:

| 输入 | 含义 | 形状(真机为例) |
|---|---|---|
| **obs** | 两路相机 RGB + 本体感觉 | 图像 `(B, 3, 216, 288)` ×2 路 + proprio |
| **action** | expert 示教的动作 chunk | **`(B, 20, 10)`** |

关键数字:**B = 256**(batch)、**H = 20**(chunk 长度)、**Da = 10**(动作维:3 平移 + 6 旋转 + 1 夹爪)、**d = 16**(潜空间维)。

## 3. 前向:一条观测,产出两份动作

观测先经 `E` 得到观测嵌入:

```
obs ──E──→ c          形状 (B, Do)        # Do = 观测嵌入维度(由 config 决定)
```

同一个 `c` 走两条平行路径,**共用同一个动作解码器 `D`**:

```
【base 路】     c ──────────────────────→ D ──→ 动作₁      条件向量 = 原嵌入 c
【探索路】     c ──detach──→ p_θ → (μ,σ) → z=μ+σε → q_ϕ → c̃ → D ──→ 动作₂   条件向量 = 还原嵌入 c̃
                               (B,16)    (B,16)   (B,Do)
```

一次前向得到两份动作预测,均为 `(B, 20, 10)`:
- **动作₁**:base 路(用原嵌入 `c` 解码)
- **动作₂**:探索路(用「压缩再还原」的嵌入 `c̃` 解码)

> 核心约束:动作₂ 若能逼近真值动作,说明 16 维潜变量保留了任务必需信息——这就是信息瓶颈。

## 4. 损失计算:4 个损失,角色分明

| Loss | 含义 | 对应论文 |
|---|---|---|
| **pred_loss** | `D(原嵌入 c)` 能否预测出 expert 动作 → base policy 标准模仿损失 | `L_IL` |
| **ext_loss** | `D(还原嵌入 c̃)` 也能预测出**同一个** expert 动作 → 逼潜变量保留任务信息 | `L_IB` 第一项(diffusion 项) |
| **kl_loss** | 潜变量分布 `(μ,σ)` 接近标准正态 → 紧凑、塑造流形 | `L_IB` 的 β·KL |
| **recon_loss** | `c̃` 是否 ≈ `c` → 额外稳定项(**默认关闭**,权重=0) | 论文中无 |

- `ext_loss` 与 `pred_loss` 算法完全相同(diffusion 去噪 MSE),唯一区别是条件向量不同(`c̃` vs `c`)。
- **`ext_loss` 是 VIB 的灵魂**:把「压缩-还原」与「能否还原出正确动作」绑定。
- 总损失:`loss = pred_loss + ext_loss·w_ext + kl_loss·β + recon_loss·w_recon`

### L_IL 与 L_IB 的具体计算（噪声预测 MSE）

> 这一节回答：论文把 L_IB 第一项写成 $-\log q_\phi(a\mid z)$（负对数似然），它怎么变成一个能优化的数。

**核心事实：$q_\phi(a\mid z)$ 是一个以 $z$ 为条件的扩散模型，不是解析分布，无法直接代入求 log。** 故 $-\log q_\phi(a\mid z)$ 用 DDPM 变分上界近似。

**L_IL（= pred_loss，base 路径，条件 = c）**
1. 给专家动作 $a$ 在随机时间步 $t$ 加噪：$a_t = \sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\varepsilon$
2. $D$ 看带噪动作 $a_t$ + 条件 $c$，预测噪声：$\hat\varepsilon = D(a_t, t, c)$
3. $\mathcal L_{IL} = \mathrm{MSE}(\hat\varepsilon,\,\varepsilon)$

**L_IB 第①项 $-\log q_\phi(a\mid z)$（= ext_loss，探索路径，条件 = c̃）**

由 DDPM 变分上界（Ho et al. 2020）：
$$-\log q_\phi(a\mid z) \;\le\; \mathbb{E}_{t\sim\mathcal U(1,T),\,\varepsilon\sim\mathcal N(0,I)}\!\Big[\big\|\varepsilon - \hat\varepsilon_\phi(a_t,\,t,\,z)\big\|^2\Big] + C$$

- $z$ 先经 $q_\phi$ 还原成条件 $\tilde c$；$z\to\tilde c$ 作条件喂给共享 $D$
- $\hat\varepsilon_\phi(a_t,t,z)$ = $D$ 在条件 $\tilde c$ 下预测的噪声
- $C$ 为与参数无关的常数，丢掉
- Monte Carlo 近似（每条动作只抽一个 $t$）：$-\log q_\phi(a\mid z) \approx \big\|\varepsilon - \hat\varepsilon_\phi\big\|^2$

> 结论：第①项落到实现上，与 L_IL 是**同一个扩散去噪 MSE**，唯一区别是条件 $c\to\tilde c$。这正是它叫 `L_IB(IL)` 的原因。

**L_IB 第②项 $\beta\,KL[p_\theta(z\mid o)\,\|\,r(z)]$（= kl_loss）**

两个对角高斯的解析 KL（后验 $p_\theta(z\mid o)=\mathcal N(\mu,\sigma^2)$，先验 $r(z)=\mathcal N(0,I)$）：
$$\beta\,KL = -\tfrac{\beta}{2}\sum_{i=1}^{d}\Big(1 + \log\sigma_i^2 - \mu_i^2 - \sigma_i^2\Big)$$
（实现里存的是 $\text{logvar}=\log\sigma^2$，即 $\sigma^2=e^{\text{logvar}}$。）

## 5. 梯度反传:每个网络只被特定的 loss 更新

如果无脑 `loss.backward()`,`ext_loss` 会顺着共用的 `D` 反传,把已训好的 base policy 带偏。SOE 用**两个手段精确隔离梯度**,使每个网络只收特定损失的梯度(✅ = 更新,– = 不更新):

| 网络 \ Loss | pred_loss | ext_loss | kl_loss | recon_loss |
|---|:---:|:---:|:---:|:---:|
| **E** 观测编码器 | ✅ | – | – | – |
| **D** 动作解码器 | ✅ | – | – | – |
| **p_θ** 压缩器 | – | ✅ | ✅ | ✅ |
| **q_ϕ** 还原器 | – | ✅ | – | ✅ |

结论:
- **`E、D` 只被 pred_loss 训练** → base policy 完全不受插件影响。
- **`p_θ、q_ϕ` 只被 ext/kl/recon 训练** → 插件不碰 base policy。

### 两个隔离手段

| 手段 | 作用对象 | 效果 |
|---|---|---|
| **手段① detach** | 探索路输入 `c.detach()` | 阻断 ext/kl/recon 的梯度回到 `E` → `E` 只收 pred_loss |
| **手段② 反传时冻结 D** | 反 ext/kl/recon 前,临时把 `D` 的 `requires_grad` 关掉 | 梯度仍能**穿过** `D` 到达 `q_ϕ`,但**不更新** `D` 自身 |

执行顺序:先反 pred_loss(给 E、D 存好梯度)→ 冻结 D → 反 ext/kl/recon(只给 p_θ、q_ϕ)→ 解冻 D。

## 6. 推理与探索

- **正常执行**:走 base 路(`D` 看 `c`)。
- **探索**:在 16 维潜变量上做 `z = μ + σ·ε·α` 的随机抖动(α = noise_scale 控制幅度),再 `q_ϕ → D` 出动作。抖动发生在「装满任务信息的潜空间」而非动作空间,故生成的是多样但仍然合理(on-manifold)的动作。
- 训练时**不使用** α(用标准重参数化 `z = μ + σ·ε`),α 仅推理探索时生效。

---
---

# 第二部分:工作流 → 代码对应关系

> 主文件:`src/policy/dp_ext.py`(类 `DPExt`)。所有行号基于该文件,除非另行注明。

## A. 四个网络在代码中的位置

| 工作流符号 | 代码模块 | 代码位置 |
|---|---|---|
| **E**(观测编码器) | `self.img_encoder`(`MultiImageObsEncoder`,内含 ResNet-18)+ 可选 `self.bottleneck` | `dp_ext.py:46-59`(`img_encoder`/`bottleneck` 构建)<br>`dp_ext.py:110-115`(前向计算 `readout`) |
| **D**(动作解码器) | `self.action_decoder`(`DiffusionUNetPolicy`) | `dp_ext.py:86`(构建)<br>类定义:`src/policy/diffusion.py:31` |
| **p_θ**(压缩器) | `self.extension_down_module`(`EncoderMLP`,输出 `style_dim×2`) | `dp_ext.py:72-76` |
| **q_ϕ**(还原器) | `self.extension_up_module`(`EncoderMLP`) | `dp_ext.py:77-81` |

> 注:`EncoderMLP` 定义在 `src/policy/vqvae_modules/vqvae.py:12`。
> `style_dim`(=d=16)、`predict_gaussian`、`kl_weight`(=β)、`ext_loss_weight`、`recon_loss_weight`、`use_mu_in_recon` 均为 `DPExt.__init__` 参数(`dp_ext.py:28-94`)。

## B. 前向流程的代码对应(`dp_ext.py: forward`)

| 工作流步骤 | 代码位置 | 说明 |
|---|---|---|
| `c = E(obs)` | `dp_ext.py:110-115` | `readout = img_encoder(obs_dict)`,经 bottleneck |
| `c.detach() → p_θ → (μ,σ)` | `dp_ext.py:118-122` | `extension_down_module(readout.detach())` 后 `chunk(2)` 得 `mu, logvar`;`std = exp(0.5·logvar)` |
| 训练时重参数化 `z = μ + σ·ε` | `dp_ext.py:127-128` | **训练分支**,不含 α |
| 探索时采样 `z = μ + σ·ε·α` | `dp_ext.py:131-143` | **推理探索分支**,乘 `self.noise_scale`(=α),可用 `std_mask` 按维掩码 |
| `c̃ = q_ϕ(z)` | `dp_ext.py:149` | `extension_up_module(readout_from_ext)` |
| (推理探索时)融合 `c` 与 `c̃` | `dp_ext.py:152-153` + `96-98` | `fuse_readout`:默认 `c + 1.0·(c̃ − c)` = 完全用 `c̃` |
| 动作解码 `D` | `dp_ext.py:196` / `diffusion.py:215` | `action_decoder.predict_action(readout)` |

## C. 四个损失的代码对应(`dp_ext.py: forward` 训练分支)

| Loss | 代码位置 | 计算方式 |
|---|---|---|
| **pred_loss** | `dp_ext.py:164-166` | `action_decoder.compute_weighted_loss(readout, actions, ...).mean()` |
| **ext_loss** | `dp_ext.py:168-170` | `action_decoder.compute_weighted_loss(readout_from_ext, actions, ...).mean()` |
| **recon_loss** | `dp_ext.py:172-179` | `mean((readout.detach() − readout_from_ext)²)`;`use_mu_in_recon=True` 时用 `μ` |
| **kl_loss** | `dp_ext.py:181-184` | `mean(−0.5·Σ(1 + logvar − μ² − exp(logvar)))`(KL → N(0,I)) |
| 总 loss 组合 | `dp_ext.py:186` | `pred_loss + ext_loss·w_ext + kl_loss·β + recon_loss·w_recon` |
| 返回字典 | `dp_ext.py:187-193` | `{"loss", "pred_loss", "ext_loss", "kl_loss", "recon_loss"}` |

> `compute_weighted_loss`(diffusion 去噪 MSE)定义在 `src/policy/diffusion.py:346-415`:对 `actions (B,20,10)` 加噪、用 `ConditionalUnet1D` 预测噪声、求 MSE。

## D. 梯度隔离(自定义 `backward`)的代码对应(`dp_ext.py:199-212`)

| 工作流步骤 | 代码位置 | 说明 |
|---|---|---|
| 先反 pred_loss | `dp_ext.py:200` | `loss["pred_loss"].backward(retain_graph=True)` → 给 E、D 存梯度 |
| **手段② 冻结 D** | `dp_ext.py:201-205` | 遍历 `action_decoder` 参数,保存原状态后 `requires_grad=False` |
| 反 ext_loss | `dp_ext.py:206` | `(ext_loss·w_ext).backward(retain_graph=True)` → 更新 p_θ、q_ϕ |
| 反 kl_loss | `dp_ext.py:207-208` | `(kl_loss·β).backward(retain_graph=True)` → 更新 p_θ |
| 反 recon_loss | `dp_ext.py:209` | `(recon_loss·w_recon).backward()` → 更新 p_θ、q_ϕ |
| **解冻 D** | `dp_ext.py:210-212` | 恢复 `action_decoder` 参数的 `requires_grad` |

> **手段① detach** 不在 `backward` 里,而在前向构建计算图时:`dp_ext.py:119` 的 `readout.detach()`。

## E. 训练循环如何触发自定义 backward

| 文件 | 位置 | 说明 |
|---|---|---|
| `src/train_single_gpu.py`(仿真,单卡) | `:183` | `loss = policy(**batch)` → 返回 loss 字典 |
| | `:194-195` | `if hasattr(policy, "backward"): policy.backward(loss)` → **直接调 DPExt.backward** |
| | `:200-201` | `optimizer.step(); optimizer.zero_grad()` |
| `src/train.py`(多卡 DDP,真机) | `:126-131` | `policy` 被 `DistributedDataParallel` 包装 |
| | `:227-228` | `hasattr(policy,"backward")` 靠 DDP 属性转发命中 → 调 `DPExt.backward` |

> 单卡路径最干净;多卡路径下「单步多次 `.backward()` + `find_unused_parameters=True`」组合较脆弱,故**仿真一律用 `train_single_gpu.py`**。

## F. SNR / 有效维度(转向用,推理阶段)

| 内容 | 代码位置 |
|---|---|
| 计算 SNR `= Var(μ)/E[σ²]` | `src/calc_snr.py:108-114` |
| 返回 `(μ, logvar)` 的接口 | `dp_ext.py:214-223`(`get_latent_action`) |
| 有效维计数(阈值 0.05) | `src/calc_snr.py:130-132` |

## G. 对照:SIME 基线(不是 SOE)

| 内容 | 代码位置 | 与 SOE 的区别 |
|---|---|---|
| SIME 噪声注入函数 | `src/policy/exploration.py:16-19` | CADS 线性衰减,直接扰 `global_cond` |
| SIME 在 diffusion 解码器内触发 | `src/policy/diffusion.py:170-185` | `enable_exploration`(≠ SOE 的 `enable_exploration_extension`) |
| SOE vs SIME 分支 | `realworld/eval.py:657-682` | `DP` 或 `DPExt+--sime` → SIME;否则 → SOE |

---

## 附:关键超参速查

| 超参 | 代码字段 | 论文/默认值 |
|---|---|---|
| 潜空间维度 d | `style_dim` | 16 |
| KL 权重 β | `kl_weight` | 1e-3(仿真) |
| 探索噪声 α | `noise_scale` | 训练默认 0.5;探索推荐 1.0–2.0 |
| ext_loss 权重 | `ext_loss_weight` | 默认 1.0 |
| recon_loss 权重 | `recon_loss_weight` | 默认 0.0(关闭) |
| batch / chunk / 动作维 | `batch_size` / `num_action` / `action_dim` | 256 / 20 / 10 |
