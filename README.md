# LaTeX OCR with Self-Supervised Pretraining and optional RL Fine-tuning

![image](https://github.com/user-attachments/assets/aba69a6d-faf1-4708-af81-c5603483b69c)


一个轻量级的 OCR 模型，用于从数学公式图像生成 LaTeX 代码。 

完美实现本地CPU推理：只要cpu不是奔腾赛扬，都能在一分钟内推理出结果，内存占用大概在0.5G左右。

以下结果是使用**纯交叉熵损失微调**后的最佳模型在**测试集**上，采用 **greedy Search ** 进行评估得到的：

| 指标 (Metric)         | 分数 (Score) |
| :-------------------- | :----------- |
| BLEU                  | 83.17        |
| Edit Distance         | 10.91        |
| Exact Match           | 30.28%       |
| Per-Token Accuracy    | 51.20%       |

*(注意: Per-Token Accuracy 计算方式为匹配的 token 数 / min(参考长度, 生成长度) 的总和)*  
后续还可训练，但设备有限，只能到这一步了.


**初始设计思路:**

1.  **阶段一：自监督预训练特征提取器 (Feature Extractor):** 使用 Transformer Encoder 架构，通过 Masked Language Model (MLM) 和结构预测（括号深度）任务进行预训练，使其理解 LaTeX 序列的结构和上下文，生成有效的特征表示。
2.  **阶段二：RL + CE 训练主 OCR 模型:** 使用标准的 Encoder-Decoder 架构（ViT Encoder + Transformer Decoder），结合交叉熵损失 (CE) 和强化学习损失 (REINFORCE)。RL 的奖励信号来源于预训练好的 Feature Extractor 计算出的生成序列与真实序列之间的特征相似度。

模型支持纯CE训练，也支持基于ExactMatch和与Feature Extractor的特征相似度来强化训练。  
本项目代码包含了完整的两阶段流程实现，用户可以通过配置文件方便地切换训练模式（纯 CE 或 CE+RL）。



# 1 数据集
本项目训练的数据集来源于IM2LATEX-100K，你需要准备包含公式图像和对应 LaTeX 源码的数据，可以去hugging face上获取。
## 1.1 目录结构
    ```
    your_repository_name/
    ├── data/
    │   ├── formula_images_processed/ # 我也不知道为什么有两个formula_images_processed，IM2LATEX-100K 源文件就是这个结构
    │   │   └── formula_images_processed/ # 最里层的才是图像的存放位置
    │   │   │  └──... (图像文件)
    │   ├── train.json            # 训练集标注 (格式见下文)
    │   ├── validate.json         # 验证集标注
    │   ├── test.json             # 测试集标注
    │   └── vocab.json            # 词汇表文件
    ├── configs/
    ├── models/
    ├── scripts/
    └── ...
    ```
## 1.2 标注文件格式 (`.json`)
    ```json
    {
      "abc.png": {
        "img_path": "formula_images_processed/abc.png", // 相对于项目根目录或绝对路径
        "caption": "\\frac{a}{b}",
        "caption_len": 5 // 可选
      },
      "def.png": {
        // ...
      }
    }
    ```
 请确保 `img_path` 指向图像文件的**正确相对路径或绝对路径**。配置文件中的 `data.image_base_dir` 用于拼接相对路径。  
 可以通过执行`scripts/prepare_data`文件构建，但是务必确保格式相同
## 1.3 词汇表 (`vocab.json`)
 需要一个包含所有 LaTeX token 到整数 ID 映射的 `vocab.json` 文件。如果缺少，你可能需要运行一个预处理脚本来从训练数据构建词汇表（但是你也可以执行 `scripts/build_vocab.py`）。确保包含特殊 token: `<PAD>`, `<SOS>`, `<EOS>`, `<UNK>`, `[CLS]`, `[SEP]`, `[MASK]`。

# 2 安装依赖
`pip install -r requirements.txt`  
运行训练或评估脚本时，如果提示缺少 `punkt`，代码会尝试自动下载。如果自动下载失败，可以手动下载。

# 3 训练
## 3.1 预训练 Feature Extractor (可选，仅用于 RL)
如果你想尝试 RL 训练，需要先预训练 Feature Extractor。

*   **配置:** 修改 `configs/pretrain_config.yaml` 文件，设置数据路径、模型参数和训练超参数。
*   **运行:**
    ```bash
    python scripts/pretrain_extractor.py --config configs/pretrain_config.yaml
    ```
*   训练好的模型检查点会保存在 `training.output_dir` 指定的目录中。
## 3.2 训练主 OCR 模型
*   **配置:** 修改 `configs/train_rl_config.yaml` 文件。
    *   设置数据路径 (`data.*`)。
    *   设置图像基目录 (`data.image_base_dir`)。
    *   设置 OCR 模型参数 (`model.*`)。
    *   **选择训练模式:**
        *   **纯交叉熵 (推荐):** 设置 `rl.lambda_rl: 0.0`。
        *   **CE + RL (实验性):**
            *   设置 `rl.lambda_rl` 为一个大于 0 的值 。
            *   在 `feature_extractor` 部分设置预训练模型的路径 (`checkpoint_path`) 和参数。
            *   设置 `rl.exact_match_bonus` 。
    *   设置训练超参数 (`training.*`)，如 `epochs`, `batch_size`, `lr` 等。
    *   **恢复训练:** 如果需要从之前的检查点恢复，设置 `training.resume_checkpoint_path` 指向 `.pth.tar` 文件。
*   **运行:**
    ```bash
    python scripts/train_rl.py --config configs/train_rl_config.yaml
    ```
*   训练日志会保存在 `training.log_file` 中。
*   模型检查点会保存在 `training.output_dir` 中，最佳模型（基于验证集 `evaluation.primary_metric`）会保存为 `model_best.pth.tar`。

但是注意：强化训练对显卡要求较高，训练速度慢，可能是代码需要优化

## 4 评估模型

*   **配置:** 修改 `configs/evaluate_config.yaml` 文件。
    *   设置测试集路径 (`data.test_split_file`)。
    *   **设置要评估的模型路径 (`model.checkpoint_path`)，通常指向训练得到的 `model_best.pth.tar`。**
    *   设置评估参数 (`evaluation.*`)，特别是 `generation_method` ('beam' 或 'greedy') 和 `beam_width`。
*   **运行:**
    ```bash
    python scripts/evaluate.py --config configs/evaluate_config.yaml
    ```
*   评估结果会打印到控制台并记录在 `evaluation_results/evaluate.log` 中。

## 配置文件 

所有超参数、路径和设置都通过 `configs/` 目录下的 YAML 文件进行管理：

*   `pretrain_config.yaml`: Feature Extractor 预训练配置。
*   `train_rl_config.yaml`: 主 OCR 模型训练配置 (CE 或 CE+RL)。
*   `evaluate_config.yaml`: 最终模型评估配置。

请根据你的环境和需求修改这些文件。

## 文件结构
```
├── configs/ # YAML 配置文件
│ ├── pretrain_config.yaml
│ ├── train_rl_config.yaml
│ └── evaluate_config.yaml
├── data/ # 数据集文件 (需要用户准备)
│ ├── formula_images_processed/
│ ├── train.json
│ ├── validate.json
│ ├── test.json
│ └── vocab.json
├── models/ # 模型定义、数据加载器、工具函数
│ ├── ocr_model.py
│ ├── feature_extractor.py
│ ├── dataloader_ocr.py
│ ├── dataloader_mtl.py
│ └── utils.py
├── scripts/ # 训练和评估脚本
│ ├── pretrain_extractor.py
│ ├── train_rl.py
│ └── evaluate.py
├── checkpoints/ # 保存训练过程中的模型检查点
│ ├── feature_extractor/
│ └── ocr_rl/
├── logs/ # 保存训练和评估日志
├── evaluation_results/ # 保存评估脚本的日志
├── Web_UI.py #可视化操作界面
└── requirements.txt # (建议提供) Python 依赖列表
└── README.md # 本文件

```

## TODO
- [x] Beam search
- [x] Web_UI 
- [ ] 实现并测试带有 KV 缓存的 Transformer 解码器以加速 RL 训练和推理
- [ ] 优化强化训练奖励函数
- [ ] 探索更先进的图像编码器或预训练策略
- [ ] attention 可视化


## 感谢
*   感谢 [im2latex-100k] 提供的数据。
*   感谢 PyTorch, Timm, NLTK, python-Levenshtein 等开源库。





