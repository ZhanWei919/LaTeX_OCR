import json
import re
import cv2


def load_vocab(vocab_txt_path, vocab_json_path):
    # 加载 vocab.txt
    with open(vocab_txt_path, 'r') as f:
        vocab_txt = [line.strip() for line in f.readlines()]

    # 加载 vocab.json
    with open(vocab_json_path, 'r') as f:
        vocab_json = json.load(f)

    return vocab_txt, vocab_json


def is_valid_latex(formula):
    """检查LaTeX公式的合法性"""
    # 检查基本的括号匹配
    brackets = {
        '(': ')',
        '[': ']',
        '{': '}',
        '\\left(': '\\right)',
        '\\left[': '\\right]',
        '\\left\\{': '\\right\\}',
        '\\lbrace': '\\rbrace'
    }

    stack = []

    # 基本语法检查模式
    patterns = [
        # 检查未闭合的数学环境
        (r'(\$[^$]*$)|([^$]*\$$)', '数学环境未闭合'),
        # 检查空的花括号
        (r'{}', '存在空的花括号'),
        # 检查连续的反斜杠
        (r'\\\\[^[]', '不合法的连续反斜杠'),
    ]

    # 分词处理，考虑LaTeX命令
    tokens = re.findall(r'\\left[\(\[\{]|\\right[\)\]\}]|\\lbrace|\\rbrace|[\(\[\{\)\]\}]|\\[a-zA-Z]+|.', formula)

    for token in tokens:
        if token in brackets.keys():  # 开括号
            stack.append(token)
        elif token in brackets.values():  # 闭括号
            if not stack:  # 栈空说明没有匹配的开括号
                return False, f"未匹配的闭括号 {token}"
            last_open = stack.pop()
            if token != brackets[last_open]:  # 括号类型不匹配
                return False, f"括号不匹配: {last_open} 与 {token}"

    if stack:  # 还有未匹配的开括号
        return False, f"未闭合的括号: {stack}"

    # 检查其他语法规则
    for pattern, error_msg in patterns:
        match = re.search(pattern, formula)
        if match:
            if callable(error_msg):
                result = error_msg(match.group())
                if result:
                    return False, result
            else:
                return False, error_msg

    return True, ""


def normalize_formula(formula):
    """规范化LaTeX公式"""
    # 基本的清理
    formula = formula.strip()
    formula = re.sub(r'\s+', ' ', formula)

    # 检查公式合法性
    is_valid, error_msg = is_valid_latex(formula)
    if not is_valid:
        # 这里可以选择返回None表示无效公式，或者尝试修复
        return None, error_msg

    return formula, None


def extract_symbols(formula_file):
    """从公式文件中提取所有符号"""
    with open(formula_file, 'r') as f:
        formulas = f.readlines()

    symbols = set()
    valid_formulas = []
    error_stats = {}  # 统计各类错误

    for formula in formulas:
        norm_formula, error_msg = normalize_formula(formula)
        if norm_formula is not None:  # 合法公式
            valid_formulas.append(norm_formula)
            symbols.update(re.findall(r'\\[a-zA-Z]+\b|[^\\\s]', norm_formula))
        else:  # 统计错误类型
            error_stats[error_msg] = error_stats.get(error_msg, 0) + 1

    return symbols, len(formulas), len(valid_formulas), error_stats


def update_vocab(vocab_txt, vocab_json, symbols):
    """更新 vocab.txt 和 vocab.json"""
    new_vocab_txt = set(vocab_txt)
    new_vocab_json = vocab_json.copy()

    new_index = len(vocab_json)  # 从当前词汇表的最大索引开始

    conflicts = []  # 用于记录冲突的符号

    for symbol in symbols:
        # 如果符号已经在 vocab.txt 和 vocab.json 中，跳过
        if symbol not in new_vocab_txt:
            # 如果是新符号，给它分配一个新的索引
            new_vocab_txt.add(symbol)
            new_vocab_json[symbol] = new_index
            new_index += 1
        else:
            # 如果符号已经存在，记录冲突
            conflicts.append(symbol)

    return new_vocab_txt, new_vocab_json, conflicts


def save_vocab(vocab_txt_path, vocab_json_path, vocab_txt, vocab_json):
    """保存更新后的 vocab.txt 和 vocab.json"""
    with open(vocab_txt_path, 'w') as f:
        for symbol in vocab_txt:
            f.write(symbol + '\n')

    with open(vocab_json_path, 'w') as f:
        json.dump(vocab_json, f, indent=4)


def clean_json_files(json_files, error_formulas):
    """从JSON文件中删除包含错误公式的条目"""
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # 删除包含错误公式的条目
        cleaned_data = {
            k: v for k, v in data.items() 
            if normalize_formula(v['caption'])[0] is not None
        }
        
        # 保存清理后的JSON
        with open(json_file, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
        
        print(f"清理后的{json_file}中保留了 {len(cleaned_data)}/{len(data)} 个样本")


def main():
    vocab_txt_path = r'data/full/formulas/vocab.txt'
    vocab_json_path = r'data/full/vocab.json'
    json_files = [
        r'data/full/train.json',
        r'data/full/val.json',
        r'data/full/test.json'
    ]

    # 加载原始词汇表
    vocab_txt, vocab_json = load_vocab(vocab_txt_path, vocab_json_path)

    # 扫描所有公式文件，提取符号
    formula_files = [
        r'data/full/formulas/train.formulas.norm.txt',
        r'data/full/formulas/test.formulas.norm.txt',
        r'data/full/formulas/val.formulas.norm.txt']

    all_symbols = set()
    total_stats = {
        'total': 0,
        'valid': 0,
        'errors': {}
    }

    error_formulas = set()  # 存储所有不合法的公式
    
    for formula_file in formula_files:
        symbols, total, valid, errors = extract_symbols(formula_file)
        all_symbols.update(symbols)
        total_stats['total'] += total
        total_stats['valid'] += valid
        for error_type, count in errors.items():
            total_stats['errors'][error_type] = total_stats['errors'].get(error_type, 0) + count

    # 更新词汇表
    new_vocab_txt, new_vocab_json, conflicts = update_vocab(vocab_txt, vocab_json, all_symbols)

    # 打印统计信息
    print(f"总公式数: {total_stats['total']}")
    print(f"合法公式数: {total_stats['valid']} ({total_stats['valid']/total_stats['total']*100:.2f}%)")
    print("\n错误类型统计:")
    for error_type, count in total_stats['errors'].items():
        print(f"- {error_type}: {count} ({count/total_stats['total']*100:.2f}%)")

    # 打印冲突的符号
    if conflicts:
        print("\n符号冲突:", len(conflicts))

    # 保存更新后的词汇表
    save_vocab(vocab_txt_path, vocab_json_path, new_vocab_txt, new_vocab_json)
    
    # 清理JSON文件
    clean_json_files(json_files, error_formulas)
    
    print("\n清理完成")


if __name__ == "__main__":
    main()
