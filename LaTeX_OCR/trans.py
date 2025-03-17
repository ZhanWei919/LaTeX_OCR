import json
import os
from PIL import Image
from tqdm import tqdm
import re

# 配置路径
base_dir = "data/full"
output_dir = "data/full"


def parse_latex_to_ast(latex, vocab):
    """将 LaTeX 公式解析为 AST，返回 ast_nodes 和 ast_structure"""

    # 定义一个简单的递归解析器
    def tokenize(latex):
        # 简单的分词，处理常见 LaTeX 结构
        tokens = re.findall(r'\\[a-zA-Z]+|[{}()]|\^|\+|-|\*|/|=|\w+|\d+', latex)
        return tokens

    def build_tree(tokens, start=0):
        """递归构建 AST"""
        nodes = []
        structure = []
        i = start

        while i < len(tokens):
            token = tokens[i]

            # 结束条件
            if token in [')', '}']:
                return nodes, structure, i

            # 基本节点
            node_id = vocab.get(token, vocab.get('UNKNOWN', -1))
            if node_id == -1:
                node_id = vocab.get('UNKNOWN', 0)  # 默认 UNKNOWN 为 0

            # 添加节点
            nodes.append(node_id)
            current_idx = len(nodes) - 1
            children = []

            # 处理上标
            if i + 1 < len(tokens) and tokens[i + 1] == '^':
                i += 2  # 跳过 ^
                if tokens[i] == '{':
                    sub_nodes, sub_structure, next_i = build_tree(tokens, i + 1)
                    children.extend(range(current_idx + 1, current_idx + 1 + len(sub_nodes)))
                    nodes.extend(sub_nodes)
                    structure.extend(sub_structure)
                    i = next_i
                else:
                    sub_node = vocab.get(tokens[i], vocab.get('UNKNOWN', 0))
                    nodes.append(sub_node)
                    children.append(current_idx + 1)
                    i += 1

            # 处理分式
            elif token == '\\frac':
                i += 1
                if tokens[i] == '{':
                    num_nodes, num_structure, next_i = build_tree(tokens, i + 1)
                    children.extend(range(current_idx + 1, current_idx + 1 + len(num_nodes)))
                    nodes.extend(num_nodes)
                    structure.extend(num_structure)
                    i = next_i + 1
                if tokens[i] == '{':
                    denom_nodes, denom_structure, next_i = build_tree(tokens, i + 1)
                    children.extend(
                        range(current_idx + 1 + len(num_nodes), current_idx + 1 + len(num_nodes) + len(denom_nodes)))
                    nodes.extend(denom_nodes)
                    structure.extend(denom_structure)
                    i = next_i

            # 添加结构
            structure.append(children)
            i += 1

        return nodes, structure, i

    tokens = tokenize(latex)
    ast_nodes, ast_structure, _ = build_tree(tokens)
    return ast_nodes, ast_structure


def process_split(split, vocab):
    """处理单个数据集划分（train/val/test），添加 AST"""
    # 读取匹配文件
    matching_path = os.path.join(base_dir, "matching", f"{split}.matching.txt")
    with open(matching_path, "r", encoding="utf-8") as f:
        pairs = [line.strip().split() for line in f]

    # 读取公式文件
    formulas_path = os.path.join(base_dir, "formulas", f"{split}.formulas.norm.txt")
    with open(formulas_path, "r", encoding="utf-8") as f:
        formulas = [line.strip() for line in f]

    # 处理图片数据
    dataset = {}
    image_dir = os.path.join(base_dir, "images", f"images_{split}")

    for img_name, formula_idx in tqdm(pairs, desc=f"Processing {split} set"):
        img_path = os.path.join(image_dir, img_name)
        if not os.path.exists(img_path):
            print(f"\nWarning: Missing image {img_path}")
            continue

        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"\nError opening {img_path}: {str(e)}")
            continue

        try:
            formula_idx = int(formula_idx)
            formula = formulas[formula_idx]
        except (ValueError, IndexError) as e:
            print(f"\nInvalid formula index {formula_idx} for {img_name}")
            continue

        # 生成 AST
        ast_nodes, ast_structure = parse_latex_to_ast(formula, vocab)

        dataset[img_name] = {
            "img_path": f"./data/full/images/images_{split}/{img_name}",
            "size": [width, height],
            "caption": formula,
            "caption_len": len(formula.split()) + 2,
            "ast_nodes": ast_nodes,
            "ast_structure": ast_structure
        }

    # 保存 JSON 文件
    output_path = os.path.join(output_dir, f"{split}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {split} set to {output_path}")


def generate_vocab():
    """生成词汇表文件"""
    src_vocab = os.path.join(base_dir, "formulas", "vocab.txt")
    dst_vocab = os.path.join(output_dir, "vocab.json")

    if os.path.exists(src_vocab):
        special_symbols = ['<start>', '<end>', '<pad>', 'UNKNOWN']
        with open(src_vocab, "r", encoding="utf-8") as src:
            vocab_lines = [line.strip() for line in src]
            for sym in special_symbols:
                if sym not in vocab_lines:
                    vocab_lines.insert(0, sym)
            vocab = {word: idx for idx, word in enumerate(vocab_lines)}
        with open(dst_vocab, "w", encoding="utf-8") as dst:
            json.dump(vocab, dst, ensure_ascii=False)
        print(f"\nVocab file saved to {dst_vocab}")
        return vocab
    else:
        print("\nWarning: Original vocab.txt not found")
        return {}


def main():
    # 生成词汇表
    vocab = generate_vocab()

    # 处理所有数据集划分
    for split in ["train", "val", "test"]:
        process_split(split, vocab)

    print("\nAll processing completed!")


if __name__ == "__main__":
    main()
