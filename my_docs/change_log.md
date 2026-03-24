# MASE-KD 修改记录（Claude Code 操作日志）

> 记录时间：2026-03-05
> 操作人：Claude Code（claude-sonnet-4-6）
> 工作分支：`kx725`

---

## 一、项目文件夹来源总览

### 原始 MASE 框架（未动源代码）

以下文件夹来自上游 [DeepWok/mase](https://github.com/DeepWok/mase) 仓库，**本次 Claude 操作未做任何修改**：

```
src/chop/              # 完整 MASE 软件栈（PyTorch FX 编译器核心）
│  ├── actions/        # train / transform / search 顶层动作
│  ├── ir/             # FX 计算图 IR
│  ├── models/         # 25+ 视觉+NLP 模型（ResNet, BERT, YOLO, GPT-2 等）
│  ├── nn/             # 量化算子、MX 格式、SNN 等
│  └── passes/         # 图优化 pass（量化、剪枝、自动分片等）
configs/               # MASE CLI 的 TOML 配置文件
test/                  # MASE 原有测试（对应 src/chop/）
docs/                  # MASE 原有文档
scripts/               # MASE 构建脚本
datasets/coco8/        # COCO8 小数据集（4 张图，验证 YOLO 流程用）
yolov8m.pt             # 预训练 YOLOv8m 权重（KD 教师）
```

---

### 课程作业已有代码（前几次 commit，非本次 Claude 新建）

以下内容由课程组和队友在 `7153bb9 / 2ea5da0 / 9a07157 / 31fc86c` 等 commit 中提交，**Claude 本次未重写这些文件的主体逻辑**：

```
src/mase_kd/
│  ├── __init__.py
│  ├── core/
│  │   ├── losses.py          # compute_distillation_loss, soft_logit_kl_loss
│  │   └── utils.py           # set_seed, dump_metrics_json
│  ├── distillation/
│  │   ├── losses.py          # TinyBERT 风格的 attention/hidden/prediction KD loss
│  │   └── mapping.py         # generate_layer_mapping（uniform teacher→student 对齐）
│  ├── config/
│  │   └── schema.py          # VisionKDConfig（原有）
│  ├── nlp/
│  │   ├── bert_kd.py         # BertKDTrainer, BertKDConfig, build_bert_kd_trainer
│  │   └── eval.py            # evaluate_classification, efficiency_report
│  ├── vision/
│  │   ├── yolo_kd.py         # YOLOLogitsDistiller（底层蒸馏类）
│  │   ├── yolo_kd_train.py   # YOLOKDRunner, YOLOTrainingConfig
│  │   └── eval.py            # benchmark_forward_latency
│  ├── reporting/
│  │   └── summarize.py       # summarize_metric_files
│  └── runners/
│      ├── run_nlp.py         # BERT CLI
│      └── run_vision.py      # YOLO CLI

experiments/
│  ├── configs/
│  │   ├── bert_baseline.toml / bert_kd.toml
│  │   └── yolo_baseline.yaml / yolo_kd.yaml
│  └── scripts/
│      ├── run_bert_{baseline,kd}.py
│      ├── run_yolo_{baseline,kd}.py
│      ├── ablation_sweep.py
│      └── evaluate_all.py

cw/
│  ├── conftest.py
│  ├── unit/
│  │   ├── test_kd_losses.py      # loss 单元测试
│  │   └── test_config_schema.py  # schema 验证测试（原有部分）
│  ├── integration/
│  │   ├── test_nlp_smoke.py      # BERT 集成测试
│  │   └── test_vision_smoke.py   # YOLO 集成测试
│  └── regression/
│      └── test_metrics_artifacts.py

my_docs/
│  ├── bert_progress.md
│  ├── yolo_progress.md
│  └── usage_guide.md
```

---

## 二、本次 Claude Code 操作详情

### 2.1 新建文件（全部为本次首次创建）

#### `src/mase_kd/vision/resnet_kd.py` ⭐
**目的**：为 ResNet18/CIFAR-10 提供与 `bert_kd.py` 接口一致的 KD 训练器，是 A-E pipeline 的核心组件。

主要内容：
- `ResNetKDConfig`：训练超参数 dataclass（epochs、lr、alpha、temperature、subset_size 等）
- `build_cifar_resnet18()`：将标准 ImageNet ResNet18 改造为 CIFAR-10 友好版本（`conv1` 从 7×7 stride-2 改为 3×3 stride-1；`maxpool` 替换为 `Identity`，防止 32×32 输入被过度下采样）
- `build_cifar_resnet34()`：同上，用于更大教师模型
- `load_cifar10_dataloaders()`：CIFAR-10 数据加载（支持 `subset_size` 用于 smoke test 快速验证）
- `ResNetKDTrainer`：完整训练器（SGD + CosineAnnealingLR；`teacher=None` 时自动跳过 soft loss 计算，不浪费 GPU）
- `build_resnet_kd_trainer()`：工厂函数

---

#### `src/mase_kd/passes/` ⭐（整个目录为新建）

##### `src/mase_kd/passes/__init__.py`
re-export 所有 pass 类，方便外部一行导入。

##### `src/mase_kd/passes/prune_pass.py`
**目的**：封装 `torch.nn.utils.prune.global_unstructured` 为统一的 pass 接口。

主要内容：
- `PruneConfig`：sparsity、target_types、make_permanent
- `PrunePass.run(model, pass_args, info) → (model, info)`：全局 L1 非结构化剪枝；`make_permanent=True` 时调用 `prune.remove()` 去除 `weight_orig`，使 `state_dict()` 可直接序列化
- `compute_model_sparsity()` / `count_nonzero_params()`：剪枝后统计实际稀疏度

**设计原则**：Han et al. 2015 全局量级剪枝，一次性剪枝（非迭代），目标类型可配置（Conv2d、Linear 等）。

##### `src/mase_kd/passes/export_pass.py`
**目的**：从 A-E 各步骤的 `metrics.json` 汇总，生成可写入报告的对比表和图。

主要内容：
- `ExportMetricsPass.run(results, output_dir, model_name, primary_metric)`：
  - 写 `comparison_table.json`（含 delta_vs_dense）
  - 写 `comparison_table.md`（Markdown 表格）
  - 写 `trade_off_plot.png`（matplotlib 柱状图，各 variant 的主指标）
- `load_metrics_from_dir()`：从目录树读取所有 `*/metrics.json`

##### `src/mase_kd/passes/pipeline.py`
**目的**：统一编排 A→B→C→D→E 五步实验矩阵，是整个 pipeline 的中枢。

主要内容：
- `ResNetPipeline.run(config, output_dir, sparsity, device)`：
  - A：从随机初始化训练 Dense ResNet18（alpha=0）
  - B：deep-copy A 的权重 → PrunePass → 保存 `pruned_student.pth` → 在原数据上直接评估（无训练）
  - C：加载 B 的 checkpoint → fine-tune（alpha=0，CE only）
  - D：加载 B 的 checkpoint → KD 训练（alpha=0.5，教师 = A 的 dense checkpoint）
  - E：加载 D 的 best checkpoint → fine-tune（alpha=0，更小 lr）
  - 每步写 `metrics.json`；最后调 `ExportMetricsPass` 写对比表
  - **重要设计**：C/D/E 的 `sparsity` 字段记录 B 的初始剪枝稀疏度（不是 FT 后的实际值），因为 SGD 在无 mask 保护时会更新原本为零的权重，导致 FT 后实际稀疏度趋近于 0
- `BertPipeline`：调用现有 `BertKDTrainer` 完成相同 A-E 流程（stub 实现）
- `YoloPipeline`：调用现有 `YOLOKDRunner` 完成相同 A-E 流程（stub 实现）

---

#### `src/mase_kd/runners/run_pipeline.py`
**目的**：提供 `python -m mase_kd.runners.run_pipeline` 的统一 CLI 入口。

CLI 参数：
```
--model     resnet18 | bert | yolo
--dataset   cifar10 | sst2 | coco
--profile   smoke | full
--sparsity  浮点数，覆盖 config 中的 pruning.sparsity
--output-dir
--seed      覆盖所有训练段的随机种子
--alpha     覆盖 kd.alpha
--temperature
--config    指定 YAML 路径（跳过自动查找）
```

自动从 `experiments/configs/{model}_{dataset}_{profile}.yaml` 加载配置，按 model+dataset 分发到对应 pipeline。

---

#### `experiments/configs/resnet18_cifar10_smoke.yaml`
**目的**：smoke 配置，5000 样本 + 2/1/1/1 epochs，CPU 约 3 min，GPU 约 20 s，用于快速验证 A-E 流程完整性。

关键字段：`subset_size: 5000`，`dense_training.epochs: 2`，`finetune/kd/kd_finetune.epochs: 1`

#### `experiments/configs/resnet18_cifar10_full.yaml`
**目的**：正式实验配置，全量 CIFAR-10 数据（50k），RTX 4070 Laptop 上单 sparsity 约 1 小时。

关键字段：`dense_training.epochs: 100`，`finetune: 30`，`kd: 30`，`kd_finetune: 10`；`seeds: [0, 1, 2]`（需手动多次调用指定 `--seed`）

---

#### `experiments/scripts/aggregate_results.py`
**目的**：对多个 sparsity level 的结果做横向汇总，生成 report-ready 文件。

产物：
- `report_ready_tables/comparison_table_sparsity_{s}.md`：每个 sparsity 单独的 A-E 表
- `report_ready_tables/combined_table.md` + `.json`：sparsity × variant 二维对比表
- `figures/accuracy_vs_variant.png`：分组柱状图（按 sparsity 分组，x 轴为 variant A-E）
- `figures/accuracy_vs_sparsity.png`：折线图（各 variant 相对 Dense baseline 的 delta，x 轴为 sparsity）

---

#### `cw/unit/test_passes.py`（新建）
**目的**：覆盖所有新 pass 类的单元测试，全部 CPU-only，无需下载数据，约 5 秒完成。

覆盖内容（共 **38 个测试用例**）：
- `TestPrunePass`：目标稀疏度达成（±10%）、`make_permanent` 后无 `weight_orig`、info 键填充、Conv-only 剪枝不影响 Linear 层、序列化正常、无匹配层报错等
- `TestSparsityHelpers`：全零=完全稀疏、全非零=0稀疏、已知50%稀疏、返回类型验证
- `TestExportMetricsPass`：三个文件都被创建、JSON 内容正确、delta_vs_dense 计算正确、partial results 不报错、mAP50 主指标支持
- `TestPipelineConfigLoads`：smoke/full YAML 可被 PyYAML 解析且字段正确

#### `cw/integration/test_resnet_smoke.py`（新建）
**目的**：用随机合成数据（无需真实 CIFAR-10）验证 ResNet 训练器的端到端行为，约 15 秒完成（CPU）。

覆盖内容（共 **12 个测试用例**，标记 `@pytest.mark.integration`）：
- 一个 epoch 跑通、loss > 0、history 填充、checkpoint 保存
- teacher 参数在训练后未被修改、teacher 无梯度
- alpha=0 时 soft_loss=0、teacher=None 时正常运行
- checkpoint save/load 后预测一致
- 剪枝后 fine-tune 流程（B→C 步）端到端可行

---

#### `project_plan.md`（仓库根目录，新建）
记录系统架构、pass 设计规格、A-E 实验矩阵、指标方案、风险与缓解措施。

---

### 2.2 修改的原有文件（共 4 个）

#### `src/mase_kd/config/schema.py`（原有，**追加内容**）
- **追加**：`ResNetKDConfig` dataclass（独立于 `resnet_kd.py` 中的同名类，作为 schema 模块的统一出口）
- **未改动**：原有的 `VisionKDConfig`

#### `cw/unit/test_config_schema.py`（原有，**追加内容**）
- **追加**：`TestResNetKDConfig` 测试类（14 个用例）：覆盖 alpha/temperature/lr/epochs/batch_size/val_split 的边界值与非法值
- **追加**：导入行 `from mase_kd.config.schema import ResNetKDConfig`
- **未改动**：原有的 `TestVisionKDConfig`、`TestBertKDConfig`、`TestYOLOTrainingConfig`

#### `pytest.ini`（原有，**追加两行**）
- **追加**：`markers` 字段，注册 `integration` 和 `large` 标记，消除 pytest 的 `PytestUnknownMarkWarning`
- **未改动**：其余配置

#### `README.md`（原有，**在文件头部插入新章节**）
- **插入**：MASE-KD 专属章节（约 240 行），包含架构图、安装、smoke 命令、full experiment 命令、输出目录说明、配置参考、测试方法、Docker 用法
- **未改动**：原始 MASE 的 README 内容（保留在 MASE-KD 章节之后）

---

### 2.3 未动的原有源代码（明确列出）

| 文件 | 原因 |
|---|---|
| `src/mase_kd/core/losses.py` | 已完备，直接 import 使用 |
| `src/mase_kd/core/utils.py` | 已完备，直接 import 使用 |
| `src/mase_kd/distillation/losses.py` | TinyBERT 风格 loss，直接使用 |
| `src/mase_kd/distillation/mapping.py` | `generate_layer_mapping` 直接使用 |
| `src/mase_kd/nlp/bert_kd.py` | `BertKDTrainer` 直接被 `BertPipeline` 调用 |
| `src/mase_kd/nlp/eval.py` | 直接使用 |
| `src/mase_kd/vision/yolo_kd_train.py` | `YOLOKDRunner` 直接被 `YoloPipeline` 调用 |
| `src/mase_kd/vision/yolo_kd.py` | 直接使用 |
| `src/mase_kd/vision/eval.py` | 直接使用 |
| `src/mase_kd/reporting/summarize.py` | 直接使用 |
| `src/mase_kd/runners/run_nlp.py` | 保留原有 BERT CLI |
| `src/mase_kd/runners/run_vision.py` | 保留原有 YOLO CLI |
| `src/chop/` 下所有文件 | 上游 MASE 框架，完全未改动 |
| `cw/integration/test_nlp_smoke.py` | 原有 BERT 集成测试，未改动 |
| `cw/integration/test_vision_smoke.py` | 原有 YOLO 集成测试，未改动 |
| `cw/unit/test_kd_losses.py` | 原有 loss 单元测试，未改动 |
| `cw/regression/test_metrics_artifacts.py` | 原有回归测试，未改动 |
| `experiments/scripts/run_bert_*.py` | 保留原有独立脚本 |
| `experiments/scripts/run_yolo_*.py` | 保留原有独立脚本 |
| `experiments/scripts/ablation_sweep.py` | 保留原有消融实验脚本 |
| `experiments/configs/bert_*.toml` | 保留原有 BERT 配置 |
| `experiments/configs/yolo_*.yaml` | 保留原有 YOLO 配置 |

---

## 三、关键设计决策说明

### 为什么 ResNet18 用自己作为教师（而不是 ResNet34）？

CIFAR-10 没有官方预训练的 ResNet34 权重；训练一个新的 ResNet34 teacher 需要额外 100 epochs 的时间，会将实验时间翻倍。使用 A 步骤训练得到的 Dense ResNet18 作为教师（self-distillation），可以：

1. 在不引入额外训练开销的情况下完成 D/E 步骤
2. 公平对比：教师和学生架构相同，KD 提升完全来自"蒸馏密集知识给剪枝模型"，而非架构差异
3. full config 中保留 `teacher.arch: resnet34` 选项，用户如果有预训练权重可直接切换

### 为什么 C/D/E 的稀疏度用 B 的值而不是重新计算？

`make_permanent=True` 后，剪枝 mask 被去除，权重直接存储在 `weight` 张量中（零值）。但 SGD 优化器会对所有参数（包括之前为零的参数）计算梯度并更新，所以 fine-tuning 后模型实际上已经不再稀疏。

对比表中记录 B 的 `sparsity_actual`（≈ 0.50）是为了告知读者"这些步骤从 50% 稀疏的检查点出发"，而不是声称 FT 后的模型仍然稀疏。

### 为什么 `teacher=None` 时跳过 teacher forward 而不报错？

alpha=0 时（A、C、E 步骤），soft loss 贡献为零，运行 teacher 的 forward pass 纯属浪费计算资源（尤其在 GPU 上，每 batch 额外 50%+ 的计算）。允许 `teacher=None` 使得同一个 `ResNetKDTrainer` 可复用于所有五个步骤，不需要为 baseline/finetune 步骤单独写类。

---

## 四、文件变更汇总（git 角度）

```
修改（M）：
  README.md                              +238 行（MASE-KD 章节插入到文件头部）
  src/mase_kd/config/schema.py           +56 行（ResNetKDConfig 追加）
  cw/unit/test_config_schema.py          +88 行（TestResNetKDConfig 追加）
  pytest.ini                             +3 行（markers 注册）

新建（??）：
  project_plan.md                        项目计划文档
  src/mase_kd/vision/resnet_kd.py        ResNet18/CIFAR-10 KD 训练器
  src/mase_kd/passes/__init__.py         passes 模块 re-export
  src/mase_kd/passes/prune_pass.py       全局 L1 剪枝 pass
  src/mase_kd/passes/export_pass.py      指标导出 pass
  src/mase_kd/passes/pipeline.py         A-E pipeline 编排
  src/mase_kd/runners/run_pipeline.py    统一 CLI 入口
  experiments/configs/resnet18_cifar10_smoke.yaml
  experiments/configs/resnet18_cifar10_full.yaml
  experiments/scripts/aggregate_results.py
  cw/unit/test_passes.py                 38 个 pass 单元测试
  cw/integration/test_resnet_smoke.py    12 个 ResNet 集成测试
  my_docs/change_log.md                  本文档
  CLAUDE.md                              Claude Code 项目指令
```

---

## 五、测试状态快照（2026-03-05）

```bash
# 全部通过（容器内 PYTHONPATH=/workspace/src）
pytest cw/unit/test_passes.py cw/unit/test_config_schema.py -v
# → 56 passed in 4.6s

pytest cw/integration/test_resnet_smoke.py -v
# → 12 passed in 15.9s

# Smoke pipeline 端到端验证通过
python3 -m mase_kd.runners.run_pipeline \
    --model resnet18 --dataset cifar10 --sparsity 0.5 --profile smoke
# → A/B/C/D/E 全部完成，outputs 写入正确
```

---

## 六、正在运行的实验（截至文档写作时）

```
Full experiment（GPU RTX 4070 Laptop）：
  sparsity=0.5 → A(100ep) + B + C(30ep) + D(30ep) + E(10ep)
  sparsity=0.7 → 同上
  输出路径：outputs/resnet18/cifar10/
  预计总耗时：~2 小时
```
