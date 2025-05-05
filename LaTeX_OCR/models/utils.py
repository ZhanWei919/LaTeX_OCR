import torch
import os
import shutil
import json
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# --- NLTK and Levenshtein ---
try:
    import nltk

    # Download punkt tokenizer models if not already present
    try:
        # 尝试查找 'punkt' 资源
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("NLTK 'punkt' resource not found. Downloading...")
        # 如果找不到，就尝试下载
        try:
            nltk.download('punkt')
            print("NLTK 'punkt' resource downloaded successfully.")
            # 再次尝试查找，确保下载成功且可用
            nltk.data.find('tokenizers/punkt')
        except Exception as download_e:
            print(f"Error downloading NLTK 'punkt' resource: {download_e}")
            print("BLEU score calculation might be affected or unavailable.")

    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    nltk_available = True

except ImportError:
    print("Warning: NLTK not found. BLEU score calculation will be unavailable.")
    print("Install NLTK: pip install nltk")
    nltk_available = False

try:
    import Levenshtein

    levenshtein_available = True
except ImportError:
    print("Warning: python-Levenshtein not found. Edit distance calculation will be unavailable.")
    print("Install python-Levenshtein: pip install python-Levenshtein")
    levenshtein_available = False


# --- Existing Utility Classes/Functions ---

class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0


def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str,
                    filename: Optional[str] = None,  # 明确标注 filename 可以是 None
                    best_filename: str = 'model_best.pth.tar'):
    """
    Saves model and training parameters at checkpoint.
    Saves the epoch checkpoint if `filename` is provided.
    Saves the best model checkpoint if `is_best` is True.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    saved_epoch_path = None  # 用于记录实际保存的 epoch 文件路径

    # 1. 只有在 filename 有效时才保存 epoch checkpoint
    if filename:  # 检查 filename 是否是有效的字符串 (不是 None 且非空)
        filepath = os.path.join(checkpoint_dir, filename)
        try:
            torch.save(state, filepath)
            logging.info(f"Epoch checkpoint saved to {filepath}")  # 建议使用 logger
            saved_epoch_path = filepath  # 记录保存的路径
        except Exception as e:
            logging.error(f"Error saving epoch checkpoint {filepath}: {e}", exc_info=True)  # 使用 logger 并记录 traceback

    # 2. 如果是最佳模型，保存 best checkpoint
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, best_filename)
        try:
            # 优化：如果刚刚保存了 epoch 文件，并且它就是最佳文件，可以尝试复制
            # 否则，直接保存状态到 best_filepath
            if saved_epoch_path:
                # 如果 epoch 文件已保存，复制它作为最佳文件
                shutil.copyfile(saved_epoch_path, best_filepath)
                logging.info(f"*** Best checkpoint saved (copied from epoch) to {best_filepath} ***")
            else:
                # 如果 epoch 文件未保存 (filename is None)，则直接保存状态
                torch.save(state, best_filepath)
                logging.info(f"*** Best checkpoint saved directly to {best_filepath} ***")
        except Exception as e:
            logging.error(f"Error saving/copying best checkpoint to {best_filepath}: {e}", exc_info=True)


def load_checkpoint(checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None,
                    scheduler=None, map_location: str = 'cpu') -> Dict[str, Any]:
    """ Loads model parameters (state_dict) from file_path. """
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint file not found at {checkpoint_path}")
        return {}
    try:
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        # --- 修改模型 Key 的查找逻辑 ---
        model_key = None
        if 'ocr_model_state_dict' in checkpoint:  # <--- 优先检查这个
            model_key = 'ocr_model_state_dict'
        elif 'model_state_dict' in checkpoint:
            model_key = 'model_state_dict'
        elif 'state_dict' in checkpoint:
            model_key = 'state_dict'
        # -----------------------------

        if model_key and model_key in checkpoint:
            # Handle potential DataParallel/DDP wrapper keys ('module.')
            state_dict = checkpoint[model_key]
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
            try:  # <--- 加上 try-except 更安全
                model.load_state_dict(state_dict, strict=True)
                print(f"Model state loaded from key '{model_key}'.")  # 打印使用的 key
            except Exception as e:
                print(f"Error loading model state dict from key '{model_key}': {e}")
            model.load_state_dict(state_dict, strict=True)  # Set strict=False if needed
            print("Model state loaded.")
        elif 'model' in checkpoint and isinstance(checkpoint['model'], torch.nn.Module):
            model.load_state_dict(checkpoint['model'].state_dict(), strict=True)
            print("Warning: Loaded state dict from a saved model object, not state_dict.")
        else:
            print(
                "Warning: Could not find compatible model state dict key ('ocr_model_state_dict', 'model_state_dict', 'state_dict') in checkpoint.")  # 更新警告信息

        # Load optimizer state dict
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded.")
        elif optimizer is not None:
            print("Warning: Optimizer state dict not found in checkpoint.")

        # Load scheduler state dict
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Scheduler state loaded.")
        elif scheduler is not None:
            print("Warning: Scheduler state dict not found in checkpoint.")

        other_info = {k: v for k, v in checkpoint.items() if
                      k not in [model_key, 'optimizer_state_dict', 'scheduler_state_dict', 'model']}
        print("Checkpoint loaded successfully.")
        return other_info
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return {}


# --- Sequence Conversion ---
def sequence_to_text(sequence: List[int],
                     rev_vocab: Dict[int, str],
                     vocab: Dict[str, int],  # Pass the forward vocab too
                     eos_token: str = '<EOS>',
                     pad_token: str = '<PAD>',
                     sos_token: str = '<SOS>') -> List[str]:
    """Converts a sequence of token IDs back to a list of tokens, stopping at EOS/PAD."""
    tokens = []
    # Get IDs for special tokens ONCE using the forward vocab
    try:
        eos_id = vocab[eos_token]
        pad_id = vocab[pad_token]
        sos_id = vocab[sos_token]
    except KeyError as e:
        print(f"Warning: Special token '{e}' not found in vocab during sequence_to_text setup.")
        # Assign unlikely IDs if not found, or handle error differently
        eos_id, pad_id, sos_id = -99, -99, -99

    for token_id in sequence:
        if isinstance(token_id, torch.Tensor):
            token_id = token_id.item()  # Convert tensor to int

        # Compare with special token IDs
        if token_id == sos_id:
            continue  # Skip SOS
        if token_id == eos_id or token_id == pad_id:
            break  # Stop at EOS or PAD

        # Use rev_vocab (int -> str) to get the token string
        tokens.append(rev_vocab.get(token_id, '?UNK?'))  # Use get with default

    return tokens


# --- Evaluation Metrics ---

def calculate_bleu(references_tokens: List[List[str]], hypothesis_tokens: List[str], smoothing: bool = True) -> float:
    """
    Calculates BLEU-4 score using NLTK.

    Args:
        references_tokens (List[List[str]]): List of reference token lists (can have multiple references).
        hypothesis_tokens (List[str]): Hypothesis token list.
        smoothing (bool): Whether to use smoothing (Method 4).

    Returns:
        float: BLEU-4 score (0 to 1).
    """
    if not nltk_available:
        return 0.0
    smooth_func = SmoothingFunction().method4 if smoothing else None
    try:
        # NLTK expects a list of references, even if there's only one.
        score = sentence_bleu(references_tokens, hypothesis_tokens, smoothing_function=smooth_func)
    except ZeroDivisionError:
        # Handle cases with very short hypotheses causing division by zero
        score = 0.0
    except Exception as e:
        print(f"Error calculating BLEU: {e}")
        score = 0.0
    return score


def calculate_edit_distance(reference_tokens: List[str], hypothesis_tokens: List[str]) -> int:
    """
    Calculates Levenshtein edit distance between two token sequences.

    Args:
        reference_tokens (List[str]): Reference token list.
        hypothesis_tokens (List[str]): Hypothesis token list.

    Returns:
        int: Edit distance. Returns -1 if Levenshtein is not available.
    """
    if not levenshtein_available:
        return -1  # Indicate unavailability
    try:
        distance = Levenshtein.distance(reference_tokens, hypothesis_tokens)
    except Exception as e:
        print(f"Error calculating Edit Distance: {e}")
        distance = -1  # Indicate error
    return distance


def calculate_exact_match(reference_tokens: List[str], hypothesis_tokens: List[str]) -> float:
    """
    Calculates exact match rate (1.0 if identical, 0.0 otherwise).

    Args:
        reference_tokens (List[str]): Reference token list.
        hypothesis_tokens (List[str]): Hypothesis token list.

    Returns:
        float: 1.0 for match, 0.0 for mismatch.
    """
    return 1.0 if reference_tokens == hypothesis_tokens else 0.0


# 在 utils.py 的 compute_metrics 函数内部
def compute_metrics(references_ids: List[List[int]],
                    hypotheses_ids: List[List[int]],
                    rev_vocab: Dict[int, str],
                    vocab: Dict[str, int],
                    eos_token: str = '<EOS>',
                    pad_token: str = '<PAD>',
                    sos_token: str = '<SOS>') -> Dict[str, float]:
    total_bleu = 0.0
    total_edit_distance = 0
    total_exact_match = 0
    count = 0

    if len(references_ids) != len(hypotheses_ids):
        logger.warning(f"compute_metrics: Refs 数量 ({len(references_ids)}) != Hyps 数量 ({len(hypotheses_ids)})")
        count = min(len(references_ids), len(hypotheses_ids))
        if count == 0:
            logger.warning("compute_metrics: 引用或假设列表为空，返回空结果。")
            return {"BLEU": 0.0, "EditDistance": 0.0, "ExactMatch": 0.0}
    else:
        count = len(references_ids)
        if count == 0:
            logger.warning("compute_metrics: 引用和假设列表均为空，返回空结果。")
            return {"BLEU": 0.0, "EditDistance": 0.0, "ExactMatch": 0.0}

    edit_distance_valid = True

    for i in range(count):

        ref_ids = references_ids[i]
        hyp_ids = hypotheses_ids[i]

        try:  # 包裹 token 转换
            ref_tokens = sequence_to_text(ref_ids, rev_vocab, vocab, eos_token, pad_token, sos_token)
            hyp_tokens = sequence_to_text(hyp_ids, rev_vocab, vocab, eos_token, pad_token, sos_token)
        except Exception as e:
            logger.error(f"compute_metrics: sequence_to_text 转换失败，样本 {i}: {e}", exc_info=True)
            continue  # 跳过这个样本

        # BLEU Score
        try:  # 包裹 BLEU 计算
            bleu_score = calculate_bleu([ref_tokens], hyp_tokens)
            total_bleu += bleu_score
        except Exception as e:
            logger.error(f"compute_metrics: calculate_bleu 计算失败，样本 {i}: {e}", exc_info=True)

        # Edit Distance
        if levenshtein_available:  # 检查库是否可用
            try:  # 包裹 Edit Distance 计算
                edit_dist = calculate_edit_distance(ref_tokens, hyp_tokens)
                if edit_dist == -1:
                    edit_distance_valid = False
                else:
                    total_edit_distance += edit_dist
            except Exception as e:
                logger.error(f"compute_metrics: calculate_edit_distance 计算失败，样本 {i}: {e}", exc_info=True)
                edit_distance_valid = False  # 出错则认为无效
        else:
            edit_distance_valid = False  # 库不可用则无效

        # Exact Match
        try:  # 包裹 Exact Match 计算
            total_exact_match += calculate_exact_match(ref_tokens, hyp_tokens)
        except Exception as e:
            logger.error(f"compute_metrics: calculate_exact_match 计算失败，样本 {i}: {e}", exc_info=True)

    avg_bleu = (total_bleu / count) * 100 if count > 0 else 0.0
    avg_edit_distance = (total_edit_distance / count) if count > 0 and edit_distance_valid else -1.0
    avg_exact_match = (total_exact_match / count) * 100 if count > 0 else 0.0

    result_metrics = {
        "BLEU": avg_bleu,
        "EditDistance": avg_edit_distance,
        "ExactMatch": avg_exact_match
    }

    return result_metrics


# --- Other utilities like JSON loading/saving, logging setup can remain here ---
def load_json(path: str) -> Any | None:
    """Loads JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON file {path}: {e}")
        return None


def save_json(data: Dict, path: str):
    """Saves dictionary to JSON file."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Data saved to {path}")
    except Exception as e:
        print(f"Error saving data to {path}: {e}")


def setup_global_logger(log_file: str, level=logging.INFO):
    """Configures the global logger."""
    log_dir = os.path.dirname(log_file)
    if log_dir: os.makedirs(log_dir, exist_ok=True)
    # Remove existing handlers to prevent duplicate logs in interactive environments
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        handlers=[logging.FileHandler(log_file, encoding='utf-8'), logging.StreamHandler()]
    )
    print(f"Logger configured. Log file: {log_file}")
    return logging.getLogger()  # 返回配置好的根 logger
