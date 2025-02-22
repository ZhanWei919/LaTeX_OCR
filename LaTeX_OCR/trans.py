import json
import os
from PIL import Image
from tqdm import tqdm

# 配置路径
base_dir = "data/full"
output_dir = "data/full"


def process_split(split):
    """处理单个数据集划分（train/val/test）"""
    # 读取匹配文件
    matching_path = os.path.join(base_dir, "matching", f"{split}.matching.txt")
    with open(matching_path, "r", encoding="utf-8") as f:
        pairs = [line.strip().split() for line in f]

    # 读取公式文件
    formulas_path = os.path.join(base_dir, "formulas", f"{split}.formulas.norm.txt")
    with open(formulas_path, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f]  # 保持为原始字符串

    # 处理图片数据
    dataset = {}
    image_dir = os.path.join(base_dir, "images", f"images_{split}")

    for img_name, formula_idx in tqdm(pairs, desc=f"Processing {split} set"):
        # 构建完整图片路径
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"\nWarning: Missing image {img_path}")
            continue

        # 获取图片尺寸
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"\nError opening {img_path}: {str(e)}")
            continue

        # 获取对应公式
        try:
            formula_idx = int(formula_idx)
            formula = formulas[formula_idx]
        except (ValueError, IndexError) as e:
            print(f"\nInvalid formula index {formula_idx} for {img_name}")
            continue

        # 构建数据条目
        dataset[img_name] = {
            "img_path": f"./data/full/images/images_{split}/{img_name}",
            "size": [width, height],
            "caption": formula,  # 保持为原始字符串
            "caption_len": len(formula.split()) + 2  # 包含<start>和<end>
        }

    # 保存JSON文件
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {split} set to {output_path}")


def generate_vocab():
    """生成词汇表文件"""
    src_vocab = os.path.join(base_dir, "formulas", "vocab.txt")
    dst_vocab = os.path.join(output_dir, "vocab.json")

    if os.path.exists(src_vocab):
        # 添加三个必需的特殊符号
        special_symbols = ['<start>', '<end>', '<pad>', 'UNKNOWN']  # 添加一个UNKNOWN符号

        with open(src_vocab, "r", encoding="utf-8") as src:
            vocab_lines = [line.strip() for line in src]

            # 确保不重复添加
            for sym in special_symbols:
                if sym not in vocab_lines:
                    vocab_lines.insert(0, sym)

            # 创建字典映射
            vocab = {word: idx for idx, word in enumerate(vocab_lines)}

        with open(dst_vocab, "w", encoding="utf-8") as dst:
            json.dump(vocab, dst, ensure_ascii=False)
        print(f"\nVocab file saved to {dst_vocab}")
    else:
        print("\nWarning: Original vocab.txt not found")


def label_transform(text, vocab):
    """转换公式标签，处理缺失的符号"""
    # 将公式中的符号映射为索引，如果缺失则使用'UNKNOWN'
    return [vocab.get(symbol, vocab.get('UNKNOWN', -1)) for symbol in text.split()]


if __name__ == "__main__":
    # 处理所有数据集划分
    for split in ["train", "val", "test"]:
        process_split(split)

    # 生成词汇表文件
    generate_vocab()

    print("\nAll processing completed!")
