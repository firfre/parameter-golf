# 动态路由实验文档

## 背景

H-net 论文的核心是动态分层 tokenization：用 RoutingModule 学习边界，不同于我们主线的固定步长方案。动态路由理论上更灵活，但在 byte260 + 短训练预算下收敛困难。

**当前固定步长最好结果**：val_bpb = 1.5629（m2-T4-m2 + Muon，80min×1H100 ≈ 竞赛 8H100×600s）

---

## 核心机制

### RoutingModule
- 计算相邻 token 的余弦相似度
- 相似度低 → 边界（boundary）
- 梯度通过 STE（Straight-Through Estimator）反传

### 关键参数
```
ROUTING_DOWNSAMPLE=8     目标压缩比，boundary_mean 目标 = 1/8 = 0.125
ROUTING_LOSS_WEIGHT=1.0  load balancing loss 权重
boundary_mean            当前边界比例（目标 0.125）
routing_aux              load balancing loss 值（≈1.0 表示正常）
```

### routing_aux 含义
- ≈ 1.0：routing 接近目标分布
- \> 1.0：边界比预期多
- \< 1.0：边界比预期少

---

## 实验记录

| 架构 | 脚本 | LR 倍率 | val_bpb | 大小 | 训练时长 | boundary_mean | 结论 |
|------|------|---------|---------|------|---------|--------------|------|
| m2-T4-m2 | v4 | — | 3.485 | 13.3MB | 10min | ~0.31（卡住） | encoder 太浅，routing 不收敛 |
| m2-T4-m2 routing_loss=3.0 | v4 | — | 6.29 | 14.8MB | 10min | — | 调坏了 |
| m4-T4-m2 + Muon | v5 | — | 2.8475 | 18.69MB | 2h | 收敛 ~0.125 | 超限；routing 收敛但 bpb 仍差 |
| m4-T4-m2 | v6 | OUTER=3.0,INNER=1.7 | 3.3820 | 17.2MB | 10min | ~0.17 未收敛 | 超限 |
| 2stage: m1-[T1m1-T2-m1T1]-m1 | v7 | OUTER=2.0,MID=1.3,INNER=0.9 | 3.8230 | 4.68MB | 10min | ~0.35 未收敛 | 模型很小但 bpb 差 |
| 2stage: m2-[T1m2-T4-m2T1]-m2 (dim=512/512/640) | v7 | OUTER=2.0,MID=1.3,INNER=0.9 | 3.2606 | 26.57MB | 2h | — | 超限；bpb 仍差 |

---

## 关键发现

**1. Encoder 深度是收敛关键**
- m2 encoder：boundary_mean 卡在 0.31，无法收敛
- m4 encoder：2000步后收敛到 0.13，趋向目标

**2. 收敛后 bpb 改善有限**
- m4-T4-m2 routing 收敛后，bpb 每1000步只降 ~0.05
- 2h 跑完 bpb 2.85，远不如固定步长 1.56

**3. 模型太大 + 无 Muon = 超限**
- 动态路由模型不能用 Muon（会报非2D参数错误）
- 无 Muon 压缩比约 2x，容易超 16MB
- 固定步长 + Muon 压缩比约 3x，同参数量节省更多空间

**4. 2-stage 更难收敛**
- 两层 routing 都需要独立收敛
- 小模型（m1层）routing 信号太弱，boundary_mean 卡在 0.35
- 大模型（m2层）2h 后 bpb 3.26，未见明显优势

---

## 脚本说明

| 脚本 | 说明 |
|------|------|
| `train_hnet_v5.py` | 固定步长或动态路由 + Muon（Muon 对动态路由有参数兼容问题） |
| `train_hnet_v6.py` | 官方 h-net 代码 + ENCODER_ARCH 字符串 + paper_group_params |
| `train_hnet_v7.py` | v6 基础上加 TWO_STAGE 支持（3层嵌套 H-net） |

---

## 下一步建议

1. **动态路由 + Muon 兼容性修复**：只对严格 2D 参数用 Muon，让动态路由也能享受更好的压缩比
2. **ROUTING_DOWNSAMPLE=6**：如果 routing 收敛困难，降低目标压缩比更容易收敛
3. **2-stage 加深 encoder**：外层 encoder 用 m4 而不是 m1/m2，给第一层 routing 更好的信号
4. **两阶段训练**：先固定步长预训练，再 fine-tune 开启动态路由

---

## 复现命令

### 单层动态路由（m4-T4-m2，2小时）
```bash
ENCODER_LAYERS=4 CHUNK_LAYERS=4 DECODER_LAYERS=2 \
VOCAB_SIZE=260 TOKENIZER_PATH=./data/tokenizers/fineweb_pure_byte_260.json \
DATA_PATH=./data/datasets/fineweb_byte260 \
MAMBA_STATE=32 MODEL_DIM=512 CHUNK_MODEL_DIM=576 CHUNK_FFN_DIM=2304 \
NUM_HEADS=8 CHUNK_ROTARY_DIM=48 \
ROUTING_DOWNSAMPLE=8 ROUTING_LOSS_WEIGHT=1.0 \
OUTER_LR_MULT=2.0 INNER_LR_MULT=1.7 \
MAX_WALLCLOCK_SECONDS=7200 VAL_LOSS_EVERY=1000 \
torchrun --standalone --nproc_per_node=1 train_hnet_v6.py
```

### 2-stage 动态路由（小模型验证）
```bash
TWO_STAGE=1 MODEL_DIM=256 MIDDLE_MODEL_DIM=256 CHUNK_MODEL_DIM=384 \
ENCODER_ARCH=m1 DECODER_ARCH=m1 \
MIDDLE_ENCODER_ARCH=T1m1 MIDDLE_DECODER_ARCH=m1T1 \
CHUNK_LAYERS=2 VOCAB_SIZE=260 \
OUTER_LR_MULT=2.0 MIDDLE_LR_MULT=1.3 INNER_LR_MULT=0.9 \
MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=100 \
torchrun --standalone --nproc_per_node=1 train_hnet_v7.py
```
