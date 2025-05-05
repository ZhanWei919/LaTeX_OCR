import torch
import torch.nn as nn  # Used for model definition import
from torch.utils.data import DataLoader
from torchvision import transforms
import json
import os
import time
import logging
import yaml
import argparse
import random
import numpy as np
from tqdm import tqdm
import sys
import traceback
from typing import Tuple, Optional, List, Dict, Any

# --- 项目路径设置 ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"添加到 sys.path: {project_root}")

# --- 导入自定义模块 ---
from models.ocr_model import OCRModel
from models.dataloader_ocr import OCRDataset, tokenize_latex, collate_ocr
from models.utils import (
    setup_global_logger, load_checkpoint,
    compute_metrics, sequence_to_text  # Import necessary utils
)

# --- 获取 logger 实例 ---
# Ensure logger is configured before use in utils or here
# We'll configure it properly in the main function
logger = logging.getLogger(__name__)


# --- 主评估函数 ---
def evaluate(config_path: str):
    # --- 1. 加载配置 ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("评估配置加载成功。")
    except Exception as e:
        print(f"错误: 无法加载评估配置文件 {config_path}: {e}")
        traceback.print_exc()
        return

    # --- 2. 设置输出目录和日志 ---
    # Use a dedicated log file for evaluation
    eval_output_dir = config.get("evaluation", {}).get("output_dir", os.path.join(project_root, "evaluation_results"))
    log_file = os.path.join(eval_output_dir, "evaluate.log")
    os.makedirs(eval_output_dir, exist_ok=True)
    # Setup logger using the utility function AFTER loading config
    logger = setup_global_logger(log_file)  # Re-assign logger with proper config
    logger.info(f"评估配置从 {config_path} 加载。")
    logger.info(f"评估结果将保存到日志: {log_file}")
    logger.info(f"项目根目录: {project_root}")

    # --- 3. 设置随机种子 (可选，但保持一致性) ---
    seed = config.get("training", {}).get("seed", 42)  # Get seed from training section or default
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"全局随机种子设置为: {seed}")

    # --- 4. 设置设备 ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("未检测到 CUDA 设备，将使用 CPU。")
    logger.info(f"使用设备: {device}")

    # --- 5. 加载词汇表 ---
    data_config = config["data"]
    vocab_file_rel = data_config["vocab_file"]
    vocab_file_abs = os.path.abspath(os.path.join(project_root, vocab_file_rel.lstrip('./').lstrip('../')))
    try:
        with open(vocab_file_abs, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        rev_vocab = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab)
        pad_token_id = vocab.get('<PAD>', 0)
        sos_token_id = vocab.get('<SOS>', 1)
        eos_token_id = vocab.get('<EOS>', 2)
        logger.info(f"词汇表加载: {vocab_size} tokens. PAD={pad_token_id}, SOS={sos_token_id}, EOS={eos_token_id}")
    except Exception as e:
        logger.error(f"无法加载词汇表 {vocab_file_abs}: {e}")
        traceback.print_exc()
        return

    # --- 6. 创建测试数据加载器 ---
    logger.info("创建测试数据加载器...")
    try:
        # Use simple transform for evaluation (no augmentation)
        eval_transform = transforms.Compose([
            transforms.Resize((data_config["image_height"], data_config["image_width"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_config["image_mean"], std=data_config["image_std"]),
        ])

        test_json_path = data_config["test_split_file"]
        if not os.path.isabs(test_json_path):
            test_json_path = os.path.abspath(os.path.join(project_root, test_json_path.lstrip('./').lstrip('../')))

        image_base_dir = data_config.get("image_base_dir")
        if image_base_dir and not os.path.isabs(image_base_dir):
            image_base_dir = os.path.abspath(os.path.join(project_root, image_base_dir.lstrip('./').lstrip('../')))
        logger.info(f"图像基目录: {image_base_dir}")

        test_dataset = OCRDataset(
            json_file=test_json_path, vocab=vocab, tokenizer=tokenize_latex,
            image_transform=eval_transform, max_seq_len=data_config["max_seq_len"],
            image_base_dir=image_base_dir
        )

        if len(test_dataset) == 0:
            logger.error("错误：测试集为空，请检查 JSON 文件路径和内容。")
            return

        eval_config = config.get("evaluation", {})
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=eval_config.get("eval_batch_size", 64),  # Get batch size from eval config
            shuffle=False,  # No shuffling for evaluation
            num_workers=data_config.get("num_workers", 0),  # Use data config workers
            collate_fn=collate_ocr,
            pin_memory=(device.type == 'cuda'),
            persistent_workers=(data_config.get("num_workers", 0) > 0 and device.type == 'cuda')
        )
        logger.info(f"测试数据加载器创建完成，共 {len(test_dataset)} 样本。")

    except Exception as e:
        logger.error(f"创建测试数据加载器失败: {e}")
        traceback.print_exc()
        return

    # --- 7. 初始化并加载模型 ---
    logger.info("初始化并加载模型...")
    model_config = config["model"]
    try:
        ocr_model = OCRModel(
            vocab_size=vocab_size, d_model=model_config.get("d_model", 768),
            decoder_nhead=model_config.get("decoder_nhead", 12), decoder_layers=model_config.get("decoder_layers", 6),
            decoder_dim_feedforward=model_config.get("decoder_dim_feedforward", 3072),
            decoder_dropout=model_config.get("decoder_dropout", 0.1), max_seq_len=data_config.get("max_seq_len", 256),
            pad_token_id=pad_token_id, sos_token_id=sos_token_id, eos_token_id=eos_token_id,
            vit_model_name=model_config.get("vit_model_name", 'vit_base_patch16_224.augreg_in21k'),
            vit_pretrained=model_config.get("vit_pretrained", False)  # Should be False for loading checkpoint
        ).to(device)

        checkpoint_path = model_config.get("checkpoint_path")
        if not checkpoint_path or checkpoint_path == "null":
            logger.error("错误：未在配置中指定模型检查点路径 (model.checkpoint_path)。")
            return
        checkpoint_abs = os.path.abspath(os.path.join(project_root, checkpoint_path.lstrip('./').lstrip('../')))
        if not os.path.isfile(checkpoint_abs):
            logger.error(f"错误：指定的模型检查点文件不存在: {checkpoint_abs}")
            return

        logger.info(f"加载模型权重从: {checkpoint_abs}")
        # Only load model weights for evaluation
        checkpoint_data = load_checkpoint(checkpoint_abs, ocr_model, optimizer=None, scheduler=None,
                                          map_location=device)
        if not checkpoint_data and not list(ocr_model.parameters())[0].is_cuda:  # Basic check if loading failed
            logger.error("模型权重加载失败，请检查检查点文件和 load_checkpoint 函数。")
            # return # Decide if you want to exit

        ocr_model.eval()  # Set model to evaluation mode
        logger.info("模型加载完成并设置为评估模式。")

    except Exception as e:
        logger.error(f"模型初始化或加载权重失败: {e}")
        traceback.print_exc()
        return

    # --- 8. 执行评估 ---
    logger.info("=" * 40)
    logger.info("开始在测试集上评估...")
    logger.info("=" * 40)

    all_references_ids = []
    all_hypotheses_ids = []
    total_compared_tokens = 0
    total_correct_tokens = 0

    start_time = time.time()
    with torch.no_grad():
        eval_pbar = tqdm(test_dataloader, desc="Evaluating on Test Set")
        for batch in eval_pbar:
            if not batch: continue
            images = batch["image"].to(device, non_blocking=True)
            # Get reference IDs as list of lists from the batch
            caption_ids_list = batch.get("caption_ids_list")
            if caption_ids_list is None:
                logger.error("Dataloader did not provide 'caption_ids_list'. Cannot evaluate.")
                break  # Stop evaluation

            try:
                # Generate predictions using the specified method
                # No need for autocast here if not using AMP during inference,
                # but keeping it won't hurt if use_amp is False in config.
                # use_amp_eval = config.get("training", {}).get("use_amp", False) # Check if AMP was used
                # amp_dtype_eval = torch.float16 if use_amp_eval and device.type == 'cuda' else None
                # with autocast(enabled=use_amp_eval, dtype=amp_dtype_eval):

                generated_ids = ocr_model.generate(
                    images,
                    max_len=data_config.get("max_seq_len", 256),
                    method=eval_config.get("generation_method", "beam"),  # Default to beam
                    beam_width=eval_config.get("beam_width", 5),
                    length_penalty=eval_config.get("length_penalty", 0.7)
                ).cpu().tolist()  # Get list of lists of IDs

                # Store for standard metrics
                all_references_ids.extend(caption_ids_list)
                all_hypotheses_ids.extend(generated_ids)

                # --- Calculate Per-Token Accuracy for this batch ---
                for ref_id_list, hyp_id_list in zip(caption_ids_list, generated_ids):
                    try:
                        # Convert IDs to tokens, automatically handles SOS/EOS/PAD
                        ref_tokens = sequence_to_text(ref_id_list, rev_vocab, vocab,
                                                      eos_token='<EOS>', pad_token='<PAD>', sos_token='<SOS>')
                        hyp_tokens = sequence_to_text(hyp_id_list, rev_vocab, vocab,
                                                      eos_token='<EOS>', pad_token='<PAD>', sos_token='<SOS>')

                        # Compare tokens up to the minimum length
                        min_len = min(len(ref_tokens), len(hyp_tokens))
                        if min_len > 0:  # Only compare if there are tokens to compare
                            correct_count = 0
                            for k in range(min_len):
                                if ref_tokens[k] == hyp_tokens[k]:
                                    correct_count += 1
                            total_compared_tokens += min_len
                            total_correct_tokens += correct_count
                        # Optional: Log sequences with 0 length if needed for debugging
                        # elif len(ref_tokens) > 0 or len(hyp_tokens) > 0:
                        #     logger.debug(f"Skipping token comparison for zero-length sequence (ref_len={len(ref_tokens)}, hyp_len={len(hyp_tokens)})")

                    except Exception as token_acc_err:
                        logger.error(f"Error calculating token accuracy for a sample: {token_acc_err}",
                                     exc_info=False)  # Log less verbosely for per-sample errors
                        continue  # Skip this sample for token accuracy

            except Exception as batch_err:
                logger.error(f"Error processing a batch during evaluation: {batch_err}", exc_info=True)
                continue  # Skip this batch

    end_time = time.time()
    eval_duration = end_time - start_time
    logger.info(f"评估循环完成，耗时: {eval_duration:.2f} 秒")

    # --- 9. 计算并报告指标 ---
    logger.info("=" * 40)
    logger.info("评估结果:")
    logger.info("=" * 40)

    # a) 标准指标
    if not all_references_ids or not all_hypotheses_ids:
        logger.warning("未收集到引用或假设，无法计算标准指标。")
        metrics = {"BLEU": 0.0, "EditDistance": -1.0, "ExactMatch": 0.0}
    elif len(all_references_ids) != len(all_hypotheses_ids):
        logger.error(f"Refs ({len(all_references_ids)}) != Hyps ({len(all_hypotheses_ids)})! 无法计算标准指标。")
        metrics = {"BLEU": 0.0, "EditDistance": -1.0, "ExactMatch": 0.0}
    else:
        try:
            logger.info("计算标准指标 (BLEU, EditDistance, ExactMatch)...")
            metrics = compute_metrics(
                references_ids=all_references_ids,
                hypotheses_ids=all_hypotheses_ids,
                rev_vocab=rev_vocab,
                vocab=vocab,
                eos_token='<EOS>',
                pad_token='<PAD>',
                sos_token='<SOS>'
            )
        except Exception as e:
            logger.error(f"计算标准指标时出错: {e}", exc_info=True)
            metrics = {"BLEU": 0.0, "EditDistance": -1.0, "ExactMatch": 0.0}  # Default on error

    logger.info("--- 标准指标 ---")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value:.4f}")

    # b) Per-Token Accuracy
    if total_compared_tokens > 0:
        token_accuracy = (total_correct_tokens / total_compared_tokens) * 100
    else:
        logger.warning("未比较任何 token，无法计算 Token Accuracy。")
        token_accuracy = 0.0

    logger.info("--- Token 级别指标 ---")
    logger.info(f"  Total Compared Tokens: {total_compared_tokens}")
    logger.info(f"  Total Correct Tokens: {total_correct_tokens}")
    logger.info(f"  Per-Token Accuracy: {token_accuracy:.4f}%")  # Display as percentage

    logger.info("=" * 40)
    logger.info("评估脚本执行完毕。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained OCR Model on the test set.")
    # Default config path relative to project root assumed by the script
    default_config_path = os.path.join(project_root, "configs", "evaluate_config.yaml")
    parser.add_argument('--config', type=str, default=default_config_path,
                        help='Path to the evaluation configuration YAML file')
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"错误: 配置文件未找到于 {args.config}")
        # Attempt fallback locations (relative to script, ../configs)
        script_dir = os.path.dirname(current_script_path)
        fallback_config_path = os.path.join(script_dir, os.path.basename(args.config))
        if os.path.exists(fallback_config_path):
            print(f"尝试使用相对路径: {fallback_config_path}")
            args.config = fallback_config_path
        else:
            fallback_config_path_2 = os.path.join(os.path.dirname(script_dir), "configs", os.path.basename(args.config))
            if os.path.exists(fallback_config_path_2):
                print(f"尝试使用 ../configs/ 路径: {fallback_config_path_2}")
                args.config = fallback_config_path_2
            else:
                sys.exit(f"配置文件在默认路径、相对路径和 ../configs 均未找到: {args.config}")

    evaluate(args.config)
