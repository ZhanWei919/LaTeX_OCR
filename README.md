![image](https://github.com/user-attachments/assets/a744965c-49e5-4daa-9726-f254e2acaaa3)# LaTeX-OCR-Small

一个轻量级的 OCR 模型，用于从数学公式图像生成 LaTeX 代码。基于 ResNet18 和 Transformer 架构，支持强化学习和对抗训练。

## 功能
- 使用 ResNet18 作为图像特征提取器，结合 Transformer 编码器和解码器生成 LaTeX 序列。
- 集成强化学习 (REINFORCE) 和对抗训练，优化生成质量。


### 环境要求
- Python >= 3.10
- CUDA

### 依赖
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.23.0
tqdm>=4.65.0
opencv-python>=4.7.0
scipy>=1.10.0
nltk>=3.8.1
python-editdistance>=1.0
Pillow>=9.5.0


### 数据集来源于
https://github.com/LinXueyuanStdio/Data-for-LaTeX_OCR
确保数据存放路径为
small
├── formulas
│   ├── train.formulas.norm.txt 规范化后的公式，以空格为分隔符直接构造字典
│   ├── test.formulas.norm.txt
│   ├── val.formulas.norm.txt
│   └── vocab.txt 根据公式文件 XXX.formulas.norm.txt 构建的字典
├── images
│   ├── images_train 图片目录
│   ├── images_test
│   └── images_val
├── matching
│   ├── train.matching.txt 样式为 <image.png>, <formulas_id> 的匹配文件
│   ├── test.matching.txt
│   └── val.matching.txt
├── data.json
├── vocab.json
└── README.md
注意：数据存在以下问题：
-部分括号可能不匹配
-vocab.txt 未包含所有字符。 建议先运行 trans.py 和 data_clean.py 补充字典


### 感谢
感谢LinXueyuanStdio所开源的模型，提供了设计思路
感谢Grok 3在debug和性能分析方面提出的建议

![image](https://github.com/user-attachments/assets/d674d3c6-e37a-4757-8e38-08ffd91e082f)
