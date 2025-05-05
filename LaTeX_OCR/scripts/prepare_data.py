import pandas as pd
import json
import os
from tqdm import tqdm
import re # 需要用到与 build_vocab 一致的清洗和分词逻辑

# --- 配置参数 ---
DATA_DIR = '../data/' # 数据根目录 (相对于脚本位置)
IMAGE_BASE_DIR = os.path.join(DATA_DIR, 'formula_images_processed', 'formula_images_processed') # 图像实际存储目录
FORMULA_FILE = os.path.join(DATA_DIR, 'im2latex_formulas.norm.csv') # 主公式文件 (可选，如果 split 文件信息足够)
SPLITS = {
    'train': 'im2latex_train.csv',
    'validate': 'im2latex_validate.csv',
    'test': 'im2latex_test.csv'
}
OUTPUT_DIR = DATA_DIR # 输出 JSON 文件到数据根目录

# --- LaTeX 清洗与分词函数 (与 build_vocab.py 保持一致) ---
def normalize_latex(formula: str) -> str:
    """对 LaTeX 公式进行基础的规范化处理"""
    if pd.isna(formula):
        return "" # 处理空值
    formula = str(formula).strip() # 确保是字符串
    formula = re.sub(r'\s+', ' ', formula)
    return formula

def tokenize_latex(formula: str) -> list[str]:
    """将规范化后的 LaTeX 公式字符串分解为 Token 列表。"""
    tokens = re.findall(r'\\(?:[a-zA-Z]+|.)|[{}]|[()\[\]]|[+\-*/=^_.,]|[a-zA-Z0-9]+', formula)
    return tokens

# --- 主逻辑 ---
def prepare_split_data():
    print("开始准备数据划分 JSON 文件...")

    # 加载词汇表，检查 <SOS>, <EOS> 是否存在 (用于计算长度)
    vocab_path = os.path.join(OUTPUT_DIR, 'vocab.json')
    if not os.path.exists(vocab_path):
        print(f"错误: 词汇表文件 {vocab_path} 未找到。请先运行 build_vocab.py。")
        return
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    if '<SOS>' not in vocab or '<EOS>' not in vocab:
        print("警告: 词汇表中未找到 <SOS> 或 <EOS> token。caption_len 计算将不包含它们。")
        len_offset = 0
    else:
        len_offset = 2 # <SOS> 和 <EOS>

    # 遍历每个数据划分 (train, validate, test)
    for split_name, split_filename in SPLITS.items():
        split_csv_path = os.path.join(DATA_DIR, split_filename)
        output_json_path = os.path.join(OUTPUT_DIR, f'{split_name}.json')

        print(f"\n处理划分: {split_name} (文件: {split_csv_path})")

        if not os.path.exists(split_csv_path):
            print(f"警告: 划分文件 {split_csv_path} 未找到，跳过此划分。")
            continue

        try:
            # 读取该划分的 CSV 文件
            df_split = pd.read_csv(split_csv_path, sep=',', dtype={'formula': str, 'image': str})
            print(f"读取 {len(df_split)} 条记录。")
        except Exception as e:
            print(f"错误: 读取 CSV 文件 {split_csv_path} 失败: {e}")
            continue

        if 'formula' not in df_split.columns or 'image' not in df_split.columns:
            print(f"错误: CSV 文件 {split_csv_path} 缺少 'formula' 或 'image' 列。")
            continue

        split_data = {}
        processed_count = 0
        skipped_count = 0

        for index, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"处理 {split_name}"):
            image_filename = row['image']
            formula = row['formula']

            if pd.isna(image_filename) or pd.isna(formula):
                # print(f"警告: 第 {index+1} 行包含空值，跳过。")
                skipped_count += 1
                continue

            # 构建图像完整路径
            image_path_relative = os.path.join(IMAGE_BASE_DIR, image_filename) # 相对于项目根目录的路径
            # 检查图像文件是否存在 (使用绝对路径或相对于此脚本的正确相对路径)
            if not os.path.exists(image_path_relative):
                 # 尝试另一种可能的相对路径（相对于 data 目录）
                 image_path_alt = os.path.join(DATA_DIR, 'formula_images_processed', 'formula_images_processed', image_filename)
                 if not os.path.exists(image_path_alt):
                      # print(f"警告: 图像文件未找到 {image_path_relative} 或 {image_path_alt}，跳过记录 {image_filename}。")
                      skipped_count += 1
                      continue
                 else:
                      image_path_to_save = image_path_alt # 使用找到的路径
            else:
                 image_path_to_save = image_path_relative # 使用第一个找到的路径

            # 清洗和分词 (确保与 build_vocab 一致)
            cleaned_formula = normalize_latex(formula)
            tokens = tokenize_latex(cleaned_formula)
            caption_len = len(tokens) + len_offset # Token 数量 + 特殊符号数量

            # 存储数据
            split_data[image_filename] = {
                # 存储相对路径，方便 Dataloader 加载
                # Dataloader 中需要知道项目根目录或 data 目录
                'img_path': os.path.relpath(image_path_to_save, start=os.path.dirname(output_json_path)), # 相对 JSON 文件的路径
                # 'img_path': image_path_to_save, # 或者存储相对于项目根的路径，需要在dataloader中处理
                'caption': cleaned_formula, # 存储清洗后的字符串
                'caption_len': caption_len
            }
            processed_count += 1

        print(f"处理完成 {processed_count} 条记录，跳过 {skipped_count} 条。")

        # 保存 JSON 文件
        print(f"保存 JSON 文件到: {output_json_path}")
        try:
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=4)
            print(f"{split_name}.json 保存成功。")
        except Exception as e:
            print(f"错误: 保存 JSON 文件 {output_json_path} 失败: {e}")

if __name__ == "__main__":
    prepare_split_data()