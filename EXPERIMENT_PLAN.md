# OT-LoRA 实验设计（COLING 2027 / ARR 2026-10-12）

**工作标题**：Decoder-side Optimal Transport Supervision for Parameter-Efficient
Fine-tuning of Medical Multimodal LLMs（暂名 OT-LoRA）

**一句话**：LoRA 微调医学 MLLM 生成报告时，在解码器隐状态与视觉 patch 特征之间加
显著性加权的最优传输辅助损失，改善临床指标与跨数据集泛化，成本仅增加 <10% 训练时间。

---

## 1. 假设（论文的三个 claim）

- **H1（设置迁移）**：LOTUS 在紧凑编码-解码器上验证的 decoder 侧 OT 监督，在
  MLLM + LoRA 低秩约束下依然有效（提升临床 F1 / 实体重叠，NLG 不掉）。
- **H2（边际设计）**：临床显著性加权边际（soft / gated）优于 uniform——修复 LOTUS
  记录的"均匀边际压低高频发现召回"的失败模式。
- **H3（泛化）**：OT 监督提升零样本跨数据集迁移（train→外部 test 集）的报告质量。

## 2. 数据（本机已确认可下载，无需 PhysioNet 凭据）

| 代号 | 来源 | 规模 | 用途 |
|---|---|---|---|
| **MIMIC-MLF** | `MLforHealthcare/mimic-cxr`（HF 镜像） | train 21,443 / val 4,594 / test 4,596，图像+findings+report | **主训练+域内评测** |
| **IU-RRG** | `dz-osamu/IU-Xray`（= R2Gen 处理版） | 7,470 图 / 3,955 报告，官方 train/val/test | 次级基准 + 跨域评测 |
| **RRG-2461** | `Yamini-1628/MIMIC-CXR-RRG`（MIMIC-CXR-RRG 官方 test 划分） | 2,461 例，findings/impression 分节 | **外部零样本评测**（与 LOTUS B.12 同源，衔接 AACL 论文） |

数据完整性说明（写论文时必须披露）：MIMIC-MLF 是社区再打包子集（非官方划分），
论文中如实描述来源与规模；RRG-2461 是 MIMIC-CXR-RRG 的官方 test。若后续拿到
PhysioNet 凭据，升级为 RRG 官方 train（~160k）重跑主实验。

**DUA 合规**：MIMIC 衍生数据不经过任何外部 API（GREEN 类 LLM 指标如需使用，
换本地 judge 模型或直接不用）。

## 3. 模型

| 角色 | 模型 | 状态 |
|---|---|---|
| 主骨干 | **Qwen2.5-VL-3B-Instruct** | 开放，hf-mirror 可下 |
| 第二骨干（规模） | Qwen2.5-VL-7B-Instruct | 开放；2×A100-80G 跑 LoRA 无压力 |
| 第二骨干（医学） | LLaVA-Rad-7B 或 MedGemma-4B | 有 gate，申请许可；失败则用 7B 顶替 |

评测器：CheXbert（14 标签）、pycocoevalcap（NLG）、自研 entity-overlap；
RadGraph-F1 视 license 获取情况（Stanford-AIMI 表单），拿到就加。

## 4. 实验矩阵

### 4.1 主表（MIMIC-MLF 训练 → 三个 test）

| 系统 | 说明 |
|---|---|
| Zero-shot | Qwen2.5-VL-3B 直接推理（prompt 工程 2-3 版取最好） |
| LoRA-SFT | 纯指令微调（baseline） |
| LoRA + ITC | + 图文对比损失（LOTUS 的辅助损失，非 OT） |
| LoRA + enc-OT | + encoder 侧特征对齐 OT（**OTDRG 式，放置点消融**） |
| **OT-LoRA (uniform)** | + decoder 侧 OT，uniform 边际（= LOTUS 平移） |
| **OT-LoRA (salience)** | + decoder 侧 OT，显著性边际（**完整方法**） |

指标：CheXbert F1-ma/mi + BLEU-1..4 / ROUGE-L / METEOR / CIDEr + distinct-2 +
entity-overlap F1。主对比跑 3 seeds（IU 上 4-6h/-run 级别，MIMIC-MLF 单 epoch
约 6-10 GPU·h，3B LoRA）；paired bootstrap 显著性。

### 4.2 H2 消融（边际设计，IU-RRG 上跑，便宜）

uniform vs soft-salience vs gated（阈值 0.5）×（balanced vs unbalanced τ=0.8）
+ ε∈{0.05,0.1,0.3}、λ_ot∈{0.02,0.05,0.1}、ratio-cap on/off、冻结视觉塔 vs 视觉塔加 LoRA。

### 4.3 分析实验（故事章节）

1. **注意力集中度**：LOTUS B.13 协议平移（有效 patch 数 / 归一化熵），zero-shot vs
   SFT vs OT-LoRA。
2. **逐标签 F1 Δ**：OT 相对 SFT 在 14 标签上的增益分布，重点看 LOTUS 里 CA-OT
   掉分的"正常模板类"标签是否被显著性边际修复（**H2 的直接证据**）。
3. **实体级幻觉率**：CheXbert 阳性标签中的错误检出比例。
4. **定性样例**：注意图 + 传输计划热图可视化（report 实体词 ↔ patch 的耦合）。

### 4.4 计划 B（Day-15 检查点，若 OT 增益全无）

切换 training-free OT 重排：同一 SFT 模型采样 k=8，OT 对齐分重排。
共享 80% 管线，2 天出表，论文改框为推理端方法。

## 5. 训练配置（起点，超参消融后定稿）

- LoRA：r=32, α=64, 目标 q/k/v/o + MLP；视觉塔冻结（默认）
- lr 1e-4（LoRA），cosine，warmup 3%，2 epochs，bs=16（grad accum）
- bf16，flash-attention2，max_len 768（图像 448² 动态分辨率）
- OT：λ=0.05，ε=0.1，30 iters，l2_mean cost，d_proj=256，ratio-cap ρ=0.5
- 显著性权重：CheXbert 离线标注 → 报告中被检为阳性/阴性的实体词 token 权重 2.0，
  其余 1.0（软）；gated 模式阈值 0.5 即只留实体词

## 6. 三周排期

| 周 | 内容 |
|---|---|
| W1（本周） | 数据管线 + salience 离线标注 + eval harness（CheXbert/NLG/distinct-n/entity）+ OT 模块单测 + Qwen3B LoRA-SFT 跑通（IU 冒烟 → MIMIC-MLF 全量） |
| W2 | 主表全配置 + 7B/第二骨干 + RRG-2461/IU 跨域评测 + Day-15 检查点 |
| W3 | 消融 + 分析实验 + 写作（骨架复用 LOTUS ≤40%，正文全新，引用 LOTUS 为先导工作） |

## 7. 风险与对策

| 风险 | 对策 |
|---|---|
| MIMIC-MLF 划分与文献常用划分不一致，审稿质疑 | 论文明示子集来源；RRG-2461 外部评测对冲；拿到凭据升级 |
| OT 增益小/不稳 | Day-15 切 Plan B；显著性检验 + 诚实 trade-off 叙事（AACL 已验证该风格可过审） |
| 第二骨干 gated 拿不到 | Qwen2.5-VL-7B 作为"规模维度"第二骨干 |
| CheXbert 权重获取 | 多个 HF 镜像可用；最坏情况用 chexpert-labeler 规则版做次级指标（LOTUS 同款，保持衔接） |
| 与 LOTUS 自我重叠质疑 | 骨干/损失形式/边际设计/指标全不同，只共享 OT 思想并正式引用 |

## 8. 当前本机状态

- 2×A100-80GB 空闲；120GB 磁盘；venv（torch 安装中）
- 数据下载中：MIMIC-MLF(792MB) → IU(1.1GB) → RRG-2461(8GB)
- 已完成：`src/ot_loss.py`（CAOTLoss + sinkhorn_log + ratio-capped weighting）
