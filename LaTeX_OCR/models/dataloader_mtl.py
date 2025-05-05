import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import re
import json
import random
from typing import Callable, List, Tuple, Dict, Any
import os  # 导入 os


# --- Tokenizer (should match build_vocab.py) ---
def tokenize_latex(formula: str) -> list[str]:
    """ Tokenizes LaTeX formula. """
    if pd.isna(formula):  # 处理 NaN 或 None
        return []
    formula = str(formula)  # 确保是字符串
    tokens = re.findall(r'\\(?:[a-zA-Z]+|.)|[{}]|[()\[\]]|[+\-*/=^_.,]|[a-zA-Z0-9]+', formula)
    return tokens


class MTLDataset(Dataset):
    """ Dataset for Multi-Task Learning (MLM + Structure Prediction) - Reads from JSON """

    def __init__(self,
                 json_file: str,  # 接收 JSON 文件路径
                 vocab: dict,
                 tokenizer: Callable[[str], List[str]],
                 max_seq_len: int = 256,
                 mlm_probability: float = 0.15,
                 ignore_index: int = -100,  # Index to ignore in MLM loss
                 max_bracket_depth: int = 10,
                 data_fraction: float = 1.0  # Optional: Use only a fraction of the data
                 ):
        """
        Args:
            json_file (str): Path to the JSON file (e.g., train.json).
            vocab (dict): Vocabulary mapping tokens to IDs.
            tokenizer (Callable): Function to tokenize a formula string.
            max_seq_len (int): Maximum sequence length for padding/truncation.
            mlm_probability (float): Probability of masking a token for MLM.
            ignore_index (int): Label value to be ignored by MLM loss function.
            max_bracket_depth (int): Maximum bracket nesting depth to record/clamp.
            data_fraction (float): Fraction of data to use (e.g., 0.1 for 10%).
        """
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.mlm_probability = mlm_probability
        self.ignore_index = ignore_index
        self.max_bracket_depth = max_bracket_depth

        # --- Load Formulas from JSON ---
        print(f"Loading formulas from JSON file: {json_file}")
        try:
            # 显式指定编码
            with open(json_file, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            # 提取所有 caption (公式)
            # 添加检查确保 'caption' 存在且不是 NaN/None
            self.formulas = [item['caption'] for item in data_dict.values() if
                             isinstance(item, dict) and 'caption' in item and pd.notna(item['caption'])]

            # Apply data fraction if specified
            if data_fraction < 1.0 and len(self.formulas) > 0:
                num_samples = int(len(self.formulas) * data_fraction)
                # Ensure num_samples is not 0 if data_fraction is very small but > 0
                num_samples = max(1, num_samples) if data_fraction > 0 else 0
                self.formulas = self.formulas[:num_samples]
                print(f"Using {data_fraction * 100:.1f}% of data: {len(self.formulas)} samples.")
            elif len(self.formulas) == 0:
                print("Warning: No valid formulas loaded from JSON file.")
            else:
                print(f"Loaded {len(self.formulas)} formulas.")

        except FileNotFoundError:
            raise IOError(f"Error: JSON file not found at {json_file}")
        except json.JSONDecodeError as e:
            raise IOError(f"Error decoding JSON file {json_file}: {e}")
        except Exception as e:
            raise IOError(f"Error loading or parsing JSON file {json_file}: {e}")

        # --- Get Special Token IDs ---
        try:
            self.cls_token_id = vocab['[CLS]']
            self.sep_token_id = vocab['[SEP]']
            self.mask_token_id = vocab['[MASK]']
            self.pad_token_id = vocab.get('<PAD>', 0)  # Use .get for safety
            self.unk_token_id = vocab['<UNK>']
            # Filter out special tokens before choosing random replacement? Optional.
            self.vocab_ids = [id for token, id in vocab.items() if
                              token not in ['[CLS]', '[SEP]', '[MASK]', '<PAD>', '<SOS>', '<EOS>']]
            if not self.vocab_ids:  # Fallback if filtering removes everything
                self.vocab_ids = list(vocab.values())

        except KeyError as e:
            raise ValueError(f"Error: Special token {e} not found in vocabulary. Please rebuild vocab.")

    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        formula = self.formulas[idx]

        # 1. Tokenize
        tokens = self.tokenizer(formula)
        # Handle potential empty list after tokenization
        if not tokens:
            tokens = [self.vocab.get('<UNK>', self.unk_token_id)]  # Use UNK if empty

        # Truncate if exceeds max_len, accounting for [CLS] and [SEP]
        max_tokens = self.max_seq_len - 2
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]

        # 2. Generate Structure Labels (Bracket Depth)
        bracket_labels = []
        current_depth = 0
        for token in tokens:
            if token in ['{', '(', '[']:
                current_depth += 1
            # Assign depth *before* decrementing for closing bracket
            bracket_labels.append(min(current_depth, self.max_bracket_depth))  # Clamp depth
            if token in ['}', ')', ']']:
                current_depth = max(0, current_depth - 1)  # Prevent negative depth

        # 3. Add Special Tokens & Convert to IDs
        input_ids = [self.cls_token_id] + [self.vocab.get(tok, self.unk_token_id) for tok in tokens] + [
            self.sep_token_id]
        # Add placeholder depth for CLS and SEP (e.g., 0)
        structure_labels = [0] + bracket_labels + [0]

        # 4. Perform MLM Masking
        mlm_labels = [self.ignore_index] * len(input_ids)  # Initialize with ignore index
        input_ids_masked = list(input_ids)  # Create a copy to modify

        for i in range(1, len(input_ids_masked) - 1):  # Iterate through tokens (excluding CLS and SEP)
            # Check if token is already special (shouldn't happen often here but good practice)
            # if input_ids_masked[i] in [self.cls_token_id, self.sep_token_id, self.pad_token_id]:
            #     continue

            if random.random() < self.mlm_probability:
                original_token_id = input_ids_masked[i]
                mlm_labels[i] = original_token_id  # Store original ID as label

                prob = random.random()
                if prob < 0.8:  # 80% Mask
                    input_ids_masked[i] = self.mask_token_id
                elif prob < 0.9:  # 10% Random Replace
                    # Ensure vocab_ids is not empty
                    if self.vocab_ids:
                        random_token_id = random.choice(self.vocab_ids)
                        input_ids_masked[i] = random_token_id
                    else:  # Fallback: keep original or mask
                        input_ids_masked[i] = self.mask_token_id  # Or keep original: pass
                # else: 10% Keep Original (implicitly handled)

        # 5. Padding
        seq_len = len(input_ids_masked)
        padding_len = self.max_seq_len - seq_len

        input_ids_padded = input_ids_masked + [self.pad_token_id] * padding_len
        attention_mask = [1] * seq_len + [0] * padding_len
        mlm_labels_padded = mlm_labels + [self.ignore_index] * padding_len
        # Pad structure labels with a value that won't contribute to loss (e.g., 0 or ignore_index if loss handles it)
        structure_labels_padded = structure_labels + [0] * padding_len  # Pad depth with 0

        # 6. Convert to Tensors
        return {
            "input_ids": torch.tensor(input_ids_padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels_padded, dtype=torch.long),
            "structure_labels": torch.tensor(structure_labels_padded, dtype=torch.long),
        }


def collate_mtl(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """ Collates batches for MTL training """
    # Stacks tensors for each key
    # Check if batch is empty
    if not batch:
        return {}
    # Get keys from the first item, assuming all items have the same keys
    keys = batch[0].keys()
    collated_batch = {key: torch.stack([item[key] for item in batch]) for key in keys}
    return collated_batch
