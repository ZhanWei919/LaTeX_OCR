import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import os
from typing import Callable, List, Tuple, Dict, Any, Optional
import logging

import re
import pandas as pd


def tokenize_latex(formula: str) -> list[str]:
    """ 对 LaTeX 公式进行分词 """
    if pd.isna(formula): return []
    formula = str(formula)
    # 正则表达式用于匹配 LaTeX 命令、特殊字符、字母数字组合等
    tokens = re.findall(r'\\(?:[a-zA-Z]+|.)|[{}]|[()\[\]]|[+\-*/=^_.,]|[a-zA-Z0-9]+', formula)
    return tokens


logger = logging.getLogger(__name__)  # 使用 logger 记录警告信息


class OCRDataset(Dataset):
    """ 用于加载图像-LaTeX 对的数据集，包含预过滤功能 """

    def __init__(self,
                 json_file: str,
                 vocab: dict,
                 tokenizer: Callable[[str], List[str]],
                 image_transform: Callable,
                 max_seq_len: int = 256,
                 image_base_dir: Optional[str] = None  # 图像文件基目录（如果 JSON 中是相对路径）
                 ):
        """
        Args:
            json_file (str): JSON 数据文件路径 (例如 train.json).
            vocab (dict): 词汇表 (token -> ID).
            tokenizer (Callable): 用于公式字符串分词的函数.
            image_transform (Callable): 应用于图像的变换函数.
            max_seq_len (int): 序列最大长度 (用于 padding).
            image_base_dir (Optional[str]): 图像文件所在的基目录.
        """
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_seq_len = max_seq_len
        self.image_base_dir = image_base_dir  # 存储基目录

        # --- 从 JSON 加载数据 ---
        logger.info(f"从 JSON 文件加载 OCR 数据: {json_file}")
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)  # 加载原始 JSON 数据
        except Exception as e:
            logger.error(f"加载或解析 JSON 文件失败 {json_file}: {e}", exc_info=True)
            raise IOError(f"加载或解析 JSON 文件时出错 {json_file}: {e}")

        # --- 预过滤数据 ---
        self.data = {}  # 存储过滤后的有效数据
        self.image_keys = []  # 存储有效数据的键（通常是图像文件名）
        skipped_count = 0  # 记录跳过的数据项数量
        logger.info("开始过滤数据集，检查有效的图像路径和标题...")
        for key, item_data in raw_data.items():
            # 检查基本格式是否正确
            if not isinstance(item_data, dict) or 'img_path' not in item_data or 'caption' not in item_data:
                logger.warning(f"跳过项 '{key}': 格式无效 - 缺少 'img_path' 或 'caption'.")
                skipped_count += 1
                continue

            img_path_from_json = item_data['img_path']  # 从 JSON 获取原始图像路径字符串
            caption = item_data['caption']  # 获取标题

            # --- 开始：路径处理逻辑 ---

            # 1. !! 关键：先将 JSON 中的路径字符串的反斜杠替换为正斜杠 !!
            #    这样可以统一路径分隔符，为后续处理做准备。
            path_with_forward_slashes = img_path_from_json.replace('\\', '/')

            # 2. 使用 os.path.basename 从 *修正了分隔符* 的路径中提取文件名
            try:
                actual_filename = os.path.basename(path_with_forward_slashes)
                # 如果 actual_filename 为空或不包含 '.', 说明原始路径可能有问题
                if not actual_filename or '.' not in actual_filename:
                    logger.warning(
                        f"跳过项 '{key}': 从 JSON 路径 '{img_path_from_json}' 提取的文件名 '{actual_filename}' 无效。")
                    skipped_count += 1
                    continue
            except Exception as e:
                logger.warning(
                    f"跳过项 '{key}': 从修正分隔符后的路径 '{path_with_forward_slashes}' 提取文件名时出错: {e}")
                skipped_count += 1
                continue

            # 3. 使用基目录和提取的 *纯文件名* 构建预期的、在当前系统上正确的路径
            if self.image_base_dir:
                # 使用 os.path.join 来确保跨平台兼容性
                img_path = os.path.join(self.image_base_dir, actual_filename)
            else:
                logger.warning("未设置 image_base_dir，尝试直接使用文件名。")
                img_path = actual_filename

            # --- 结束：再次修改路径处理逻辑 ---

            # 4. 使用 *构建好的路径* 检查图像文件是否存在
            if not os.path.exists(img_path):
                logger.warning(
                    f"跳过项 '{key}': 在构建的路径 '{img_path}' 未找到图像文件。JSON 中的原始路径是 '{img_path_from_json}'。")
                skipped_count += 1
                continue

            # 5. 检查标题是否有效（例如，非空）
            if pd.isna(caption) or not str(caption).strip():
                logger.warning(f"跳过项 '{key}': 标题无效或为空。")
                skipped_count += 1
                continue

            # --- 新增：尝试打开图像文件以检查是否损坏 ---
            try:
                # 尝试打开图像并立即关闭，验证文件是否至少是可读的图像格式
                with Image.open(img_path) as img_test:
                    img_test.verify()  # 尝试验证图像数据，可能捕获一些损坏
                # 如果上面没有抛出异常，说明文件至少可以被 PIL 识别
            except (IOError, SyntaxError, Image.DecompressionBombError, Exception) as img_err:
                # 捕获 PIL 可能抛出的各种图像错误
                logger.warning(
                    f"跳过项 '{key}': 图像文件 '{img_path}' 存在但无法打开或验证，可能已损坏或格式错误: {img_err}")
                skipped_count += 1
                continue
            # --- 结束：新增图像文件检查 ---

            # 6. 如果所有检查都通过，则将数据添加到过滤后的结果中
            self.data[key] = item_data
            self.data[key]['_resolved_img_path'] = img_path  # 存储修正后的有效路径
            self.image_keys.append(key)

        # 检查过滤后是否还有有效数据
        if not self.image_keys:
            raise ValueError(
                f"在过滤后，未能从 {json_file} 加载任何有效数据。请检查 JSON 内容、image_base_dir 设置以及图像文件本身。")

        logger.info(
            f"过滤完成。加载了 {len(self.image_keys)} 个有效的图像-标题对。跳过了 {skipped_count} 个无效项。")

        # --- 获取特殊 Token 的 ID ---
        try:
            self.sos_token = '<SOS>'
            self.eos_token = '<EOS>'
            self.pad_token = '<PAD>'
            self.unk_token = '<UNK>'
            self.sos_token_id = vocab[self.sos_token]
            self.eos_token_id = vocab[self.eos_token]
            self.pad_token_id = vocab[self.pad_token]
            self.unk_token_id = vocab[self.unk_token]
        except KeyError as e:
            logger.error(f"特殊 token {e} 在词汇表中未找到。", exc_info=True)
            raise ValueError(f"错误: 特殊 token {e} 在词汇表中未找到。")

    def __len__(self):
        """ 返回数据集中有效样本的数量 """
        return len(self.image_keys)

    def __getitem__(self, idx) -> Dict[str, Any]:
        """ 获取指定索引的数据样本 """
        image_key = self.image_keys[idx]
        item_data = self.data[image_key]
        # 使用在 __init__ 中预处理和验证过的正确路径
        img_path = item_data['_resolved_img_path']
        caption = item_data['caption']

        # --- 加载和转换图像 ---
        # 在 __init__ 中已经尝试过打开，这里理论上应该没问题，但保留 try-except 更安全
        try:
            image = Image.open(img_path).convert('RGB')
            image_tensor = self.image_transform(image)
        except Exception as e:
            logger.error(f"处理图像 {img_path} (对应项 '{image_key}', 索引 {idx}) 时出错: {e}", exc_info=True)
            raise RuntimeError(f"处理图像失败 {img_path}") from e

        # --- 对标题进行分词和 Padding ---
        tokens = self.tokenizer(caption)
        if len(tokens) > self.max_seq_len - 2:  # -2 for SOS and EOS
            tokens = tokens[:self.max_seq_len - 2]

        # 创建包含 SOS 和 EOS 的原始 ID 列表 (未 padding)
        token_ids_unpadded = [self.sos_token_id] + \
                             [self.vocab.get(tok, self.unk_token_id) for tok in tokens] + \
                             [self.eos_token_id]
        original_len = len(token_ids_unpadded)  # 实际长度

        # Padding
        padding_len = self.max_seq_len - original_len
        token_ids_padded = token_ids_unpadded + [self.pad_token_id] * padding_len

        return {
            "image": image_tensor,
            "caption_ids": torch.tensor(token_ids_padded, dtype=torch.long),  # Padded tensor
            "caption_len": torch.tensor(original_len, dtype=torch.long),  # Original length
            "caption_ids_list": token_ids_unpadded  # <-- 新增：未 padding 的 ID 列表
        }


# --- collate_ocr 函数：用于将单个样本组成的列表合并成一个批次 ---
def collate_ocr(batch: List[Dict[str, Any]]) -> Dict[str, Any]:  # 返回类型可能包含列表，改为 Any
    """
    合并 OCR 训练/验证的批次数据。
    """
    if not batch:
        logger.warning("collate_ocr 收到了一个空批次。")
        return {}

    keys = batch[0].keys()
    collated_batch = {}

    # 堆叠张量
    tensor_keys = ["image", "caption_ids", "caption_len"]
    for key in tensor_keys:
        if key in keys:  # 检查键是否存在
            try:
                collated_batch[key] = torch.stack([item[key] for item in batch])
            except Exception as e:
                # 如果某个 item 缺少这个 tensor key，stack 会失败
                logger.error(f"Collating key '{key}' failed: {e}. Batch items might be inconsistent.")
                # 可以选择返回空字典或部分批次，这里简单返回空
                return {}

    # 收集列表 (例如 caption_ids_list)
    list_keys = ["caption_ids_list"]
    for key in list_keys:
        if key in keys:
            collated_batch[key] = [item[key] for item in batch]  # 直接收集列表

    # 检查 caption_ids_list 是否真的被收集了
    if "caption_ids_list" not in collated_batch and "caption_ids_list" in keys:
        logger.warning("Key 'caption_ids_list' existed in items but was not collected.")
    elif "caption_ids_list" not in keys:
        logger.warning("Key 'caption_ids_list' was not found in the dataset items.")

    return collated_batch
