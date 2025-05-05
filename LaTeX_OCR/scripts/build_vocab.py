import pandas as pd
import re
from collections import Counter
import json
import os
from tqdm import tqdm

# --- 配置参数 ---
FORMULA_FILE = '../data/im2latex_formulas.norm.csv'  # 主公式文件路径 (相对于脚本位置)
VOCAB_FILE = '../data/vocab.json'  # 输出词汇表文件路径
MIN_FREQ = 3  # 保留 token 的最低频率
# 特殊 Tokens (确保与模型和 Dataloader 预期一致)
PAD_TOKEN = '<PAD>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'
UNK_TOKEN = '<UNK>'
# MLM/FeatureExtractor 可能需要的特殊 tokens
CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'

SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN,
                  CLS_TOKEN, SEP_TOKEN, MASK_TOKEN]


# --- LaTeX 清洗与分词函数 ---

def normalize_latex(formula: str) -> str:
    """对 LaTeX 公式进行基础的规范化处理"""
    # 1. 去除首尾空格
    formula = formula.strip()
    # 2. 将多个连续空格替换为单个空格
    formula = re.sub(r'\s+', ' ', formula)
    # 3.  在特定符号周围添加空格，方便分词
    #    例如：确保括号、运算符、上下标符号周围有空格
    #    这步比较复杂，暂时简化处理，依赖下面的分词器
    formula = re.sub(r'([\{\}\(\)\[\]\+\-\*/=_,.^])', r' \1 ', formula)
    formula = re.sub(r'\s+', ' ', formula).strip() # 再次清理空格
    return formula


def tokenize_latex(formula: str) -> list[str]:
    """
    将规范化后的 LaTeX 公式字符串分解为 Token 列表。
    规则：
    1. LaTeX 命令: \ 开头，后跟字母或单个非字母字符 (\sin, \alpha, \+)
    2. 括号和特殊符号: {} () [] + - * / = _ ^ . ,
    3. 字母数字组合: [a-zA-Z0-9]+
    """
    # 正则表达式尝试匹配上述规则
    # \\(?:[a-zA-Z]+|.) : 匹配 \command 或 \特殊字符
    # [{}] : 匹配 { 或 }
    # [()\[\]] : 匹配 ( ) [ ]
    # [+\-*/=^_.,] : 匹配常用运算符和结构符 (注意 - 和 ^ 在字符集中需要小心处理)
    # [a-zA-Z0-9]+ : 匹配一个或多个字母或数字
    tokens = re.findall(r'\\(?:[a-zA-Z]+|.)|[{}]|[()\[\]]|[+\-*/=^_.,]|[a-zA-Z0-9]+', formula)
    return tokens


# --- 主逻辑 ---
def build_vocabulary():
    print(f"开始处理公式文件: {FORMULA_FILE}")

    # 检查文件是否存在
    if not os.path.exists(FORMULA_FILE):
        print(f"错误: 公式文件未找到 {FORMULA_FILE}")
        return

    try:
        # 读取 CSV 文件，假设是制表符分隔
        df = pd.read_csv(FORMULA_FILE, sep='\t', dtype={'formula': str, 'image': str})
        print(f"成功读取 {len(df)} 条公式记录。")
    except Exception as e:
        print(f"错误: 读取 CSV 文件失败: {e}")
        print("请确认文件路径和分隔符 (sep='\t') 是否正确。")
        return

    # 检查列名是否存在
    if 'formulas' not in df.columns:
        print("错误: CSV 文件中未找到 'formula' 列。")
        return

    print("开始规范化和分词...")
    token_counts = Counter()
    formulas = df['formulas'].astype(str).tolist()  # 确保是字符串类型

    for formula in tqdm(formulas, desc="处理公式"):
        if pd.isna(formula):  # 跳过空值
            continue
        cleaned_formula = normalize_latex(formula)
        tokens = tokenize_latex(cleaned_formula)
        token_counts.update(tokens)

    print(f"完成分词。总 Token 数 (含重复): {sum(token_counts.values())}")
    print(f"独立 Token 种类数 (含特殊符号): {len(token_counts)}")

    # 构建词汇表
    print(f"根据最低频率 {MIN_FREQ} 构建词汇表...")
    vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}  # 先加入特殊 token
    current_id = len(SPECIAL_TOKENS)

    frequent_tokens = 0
    infrequent_tokens = 0
    for token, count in token_counts.items():
        if count >= MIN_FREQ:
            if token not in vocab:  # 避免重复添加特殊符号
                vocab[token] = current_id
                current_id += 1
                frequent_tokens += 1
        else:
            infrequent_tokens += 1

    # 确保 UNK token 在里面 (虽然前面加了，再确认下)
    if UNK_TOKEN not in vocab:
        vocab[UNK_TOKEN] = current_id
        current_id += 1

    print(f"词汇表构建完成:")
    print(f"  - 特殊 Tokens: {len(SPECIAL_TOKENS)}")
    print(f"  - 高频 Tokens (频率 >= {MIN_FREQ}): {frequent_tokens}")
    print(f"  - 低频 Tokens (被替换为 <UNK>): {infrequent_tokens}")
    print(f"  - 最终词汇表大小: {len(vocab)}")

    # 保存词汇表
    print(f"保存词汇表到: {VOCAB_FILE}")
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(VOCAB_FILE), exist_ok=True)
        with open(VOCAB_FILE, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=4)
        print("词汇表保存成功。")
    except Exception as e:
        print(f"错误: 保存词汇表失败: {e}")


if __name__ == "__main__":
    build_vocabulary()
