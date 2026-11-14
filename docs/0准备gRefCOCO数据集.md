# 1. 准备 gRefCOCO 数据集

gRefCOCO 是首个面向 **广义指称表达（GREx）** 的大规模基准：一句表达可指 **一个、多个或零个** 目标，同时支持 GREC（框）与 GRES（掩码）。数据集基于 COCO `train2014` 的 **19,994** 张图像构建，图像与实例标注均可直接复用 COCO。

> **速览**　94,050 个样本 · 258,836 条释义 · 121,476 条目标标注 · 约 13 GB 图像

---

## 1.1 数据集划分

**样本单位是 ref**（`ref_id` 全局唯一）；一个 ref 含多条同义释义，训练时随机取一条。下表统计自 `grefs(unc).json`：

| 子集 | refs（样本） | 图像 | 单目标 | 多目标 | 无目标 | 释义数 |
| :--- | ----------: | ---: | -----: | -----: | -----: | -----: |
| `train` | 78,930 | 16,994 | 42,404 | 25,060 | 11,466 | 209,344 |
| `val` | 4,237 | 1,501 | 0 | 2,738 | 1,499 | 14,229 |
| `testA` | 6,265 | 750 | 1,975 | 3,541 | 749 | 19,200 |
| `testB` | 4,618 | 750 | 1,807 | 2,061 | 750 | 16,063 |
| **合计** | **94,050** | **19,994** | **46,186** | **33,400** | **14,464** | **258,836** |

**说明**

- `train` 训练、`val` 选模调参、`testA` / `testB` 测试；三者图像均与 `train` 不重叠（`testA` 与 `val` 共用 1 张）。
- 无目标 ref 的 `targets` 仅有占位项：`bbox=null`、`id=-1`、`empty=true`，计算框回归 / IoU 时须跳过。
- 本版本 `val` 只含多目标与无目标 ref；`testA` / `testB` 三类齐全。
- 85 个目标带 `iscrowd=1`（分布在 85 个 ref），官方 ReLA loader 直接跳过，按需过滤。
- 数字以本地 `grefs(unc).json` 为准；不同论文引用的释义总数口径略有差异。

---

## 1.2 下载数据

### ① 标注文件（约 179 MB）

从 [HuggingFace · FudanCVL/gRefCOCO](https://huggingface.co/datasets/FudanCVL/gRefCOCO) 获取 `grefs(unc).json` 与 `instances.json`：

```bash
pip install -U huggingface_hub                  # 提供 hf / huggingface-cli
mkdir -p data/grefs
hf download FudanCVL/gRefCOCO \
  --repo-type dataset --local-dir data/grefs
```

也可从 [官方仓库](https://github.com/henghuiding/gRefCOCO) 的 Google Drive 手动下载。最终确保得到 `data/grefs/grefs(unc).json` 与 `data/grefs/instances.json`（若被下到子目录，上移一层）。

### ② COCO 图像与官方标注（约 13 GB + 241 MB）

```bash
wget -P data/grefs http://images.cocodataset.org/zips/train2014.zip
wget -P data/grefs http://images.cocodataset.org/annotations/annotations_trainval2014.zip
unzip -q data/grefs/train2014.zip -d data/grefs
unzip -q data/grefs/annotations_trainval2014.zip -d data/grefs
```

---

## 1.3 用脚本预处理

```bash
python scripts/prepare_grefcoco.py \
  --data_path data/grefs \
  --out_path  data/grefs \
  --image_root data/grefs/train2014/
```

产出 `data/grefs/grefcoco_{train,val,testA,testB}.json`，每个 ref 一条记录：

| 字段 | 内容 |
| :--- | :--- |
| `image` / `file_name` | 图像尺寸与文件名 |
| `captions` | COCO 图像描述（通常 5 条） |
| `sentences` | 多条同义释义（训练时随机取一条） |
| `targets` | `bbox`、`segmentation`、`iscrowd`、`empty` |

**常用参数**

| 参数 | 说明 |
| :--- | :--- |
| `--splits` | 只处理部分子集，如 `--splits train val` |
| `--no-strict` | 跳过坏样本而非中断 |
| `--instances_file` | 覆盖默认实例标注路径 |
| `--captions_file` | 覆盖默认 captions 路径 |

默认输入依次为 `grefs(unc).json`、`instances.json`（缺省时退回 `annotations/instances_train2014.json`）、`annotations/captions_train2014.json`；`--image_root` 会写入输出，loader 用 `image_root + file_name` 拼出图片路径。

---

## 1.4 读取与目录结构

```python
ref     = random.choice(split["refs"])
sent    = random.choice(ref["sentences"])["sent"]        # 释义增强
targets = [t for t in ref["targets"]
           if not t["empty"] and not t["iscrowd"]]
```

```text
data/grefs/
├── grefs(unc).json      # 原始标注
├── instances.json       # 19,994 张图的 COCO 实例标注
├── annotations/         # COCO 官方标注（captions 等）
├── train2014/           # COCO 图像
└── grefcoco_*.json      # 预处理产物
```
