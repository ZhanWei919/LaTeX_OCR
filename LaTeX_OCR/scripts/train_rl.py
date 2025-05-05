from typing import Tuple, Optional, List, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
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

# --- (Profiler import - optional) ---
# from torch.profiler import profile, record_function, ProfilerActivity

# --- 项目路径设置 ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"添加到 sys.path: {project_root}")

# --- 导入自定义模块 ---
from models.ocr_model import OCRModel
from models.feature_extractor import FeatureExtractorMTL  # Keep import for type hinting
from models.dataloader_ocr import OCRDataset, tokenize_latex, collate_ocr
# Feature extractor tokenizer might not be needed if FE is only used inside RL block
# from models.dataloader_mtl import tokenize_latex as fe_tokenizer
from models.utils import (
    AverageMeter, setup_global_logger, save_checkpoint, load_checkpoint,
    compute_metrics, sequence_to_text
)


# --- 学习率预热 ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Linear warmup and decay scheduler. """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# --- 主训练函数 ---
def train_rl(config_path: str):
    # --- 1. 加载配置 ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功。")
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        traceback.print_exc()
        return

    # --- 2. 设置输出目录和日志 ---
    output_dir_rel = config["training"]["output_dir"]
    log_file_rel = config["training"]["log_file"]
    output_dir = os.path.abspath(os.path.join(project_root, output_dir_rel.lstrip('./').lstrip('../')))
    log_file = os.path.abspath(os.path.join(project_root, log_file_rel.lstrip('./').lstrip('../')))
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_global_logger(log_file)
    logger.info(f"配置从 {config_path} 加载。")
    logger.info(f"输出将保存到: {output_dir}")
    logger.info(f"日志将记录到: {log_file}")
    logger.info(f"项目根目录: {project_root}")

    # --- 3. 设置随机种子 ---
    seed = config["training"]["seed"]
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
    vocab_file_rel = config["data"]["vocab_file"]
    vocab_file_abs = os.path.abspath(os.path.join(project_root, vocab_file_rel.lstrip('./').lstrip('../')))
    try:
        with open(vocab_file_abs, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        rev_vocab = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab)
        pad_token_id = vocab.get('<PAD>', 0)
        sos_token_id = vocab.get('<SOS>', 1)
        eos_token_id = vocab.get('<EOS>', 2)
        cls_token_id = vocab.get('[CLS]')  # Needed for prepare_for_feature_extractor
        sep_token_id = vocab.get('[SEP]')  # Needed for prepare_for_feature_extractor
        lambda_rl = float(config["rl"]["lambda_rl"])  # Get lambda_rl early

        # Check for CLS/SEP only if RL is enabled
        if lambda_rl > 0 and (cls_token_id is None or sep_token_id is None):
            logger.error("RL 训练需要 [CLS] 和 [SEP] tokens，但它们不在词汇表中。")
            return

        logger.info(
            f"词汇表加载: {vocab_size} tokens. PAD={pad_token_id}, SOS={sos_token_id}, EOS={eos_token_id}"
            f"{f', CLS={cls_token_id}, SEP={sep_token_id}' if lambda_rl > 0 else ''}")
    except Exception as e:
        logger.error(f"无法加载词汇表 {vocab_file_abs}: {e}")
        traceback.print_exc()
        return

    # --- 6. 创建数据加载器 ---
    logger.info("创建数据加载器...")
    data_config = config["data"]
    h, w = data_config["image_height"], data_config["image_width"]
    mean, std = data_config["image_mean"], data_config["image_std"]
    try:
        train_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=5),
            transforms.ToTensor(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.5),
            transforms.Normalize(mean=mean, std=std),
        ])
        val_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        train_json_path = data_config["train_split_file"]
        val_json_path = data_config["val_split_file"]
        if not os.path.isabs(train_json_path): train_json_path = os.path.abspath(
            os.path.join(project_root, train_json_path.lstrip('./').lstrip('../')))
        if not os.path.isabs(val_json_path): val_json_path = os.path.abspath(
            os.path.join(project_root, val_json_path.lstrip('./').lstrip('../')))
        image_base_dir = data_config.get("image_base_dir")
        if image_base_dir and not os.path.isabs(image_base_dir): image_base_dir = os.path.abspath(
            os.path.join(project_root, image_base_dir.lstrip('./').lstrip('../')))
        logger.info(f"图像基目录: {image_base_dir}")
        train_dataset = OCRDataset(json_file=train_json_path, vocab=vocab, tokenizer=tokenize_latex,
                                   image_transform=train_transform, max_seq_len=data_config["max_seq_len"],
                                   image_base_dir=image_base_dir)
        val_dataset = OCRDataset(json_file=val_json_path, vocab=vocab, tokenizer=tokenize_latex,
                                 image_transform=val_transform, max_seq_len=data_config["max_seq_len"],
                                 image_base_dir=image_base_dir)
        if len(train_dataset) == 0 or len(val_dataset) == 0: logger.error("错误：训练集或验证集为空..."); return
        train_dataloader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True,
                                      num_workers=data_config["num_workers"] if device.type == 'cuda' else 0,
                                      collate_fn=collate_ocr, pin_memory=(device.type == 'cuda'),
                                      persistent_workers=(data_config["num_workers"] > 0 and device.type == 'cuda'))
        val_dataloader = DataLoader(val_dataset, batch_size=config["evaluation"]["eval_batch_size"], shuffle=False,
                                    num_workers=data_config["num_workers"] if device.type == 'cuda' else 0,
                                    collate_fn=collate_ocr, pin_memory=(device.type == 'cuda'),
                                    persistent_workers=(data_config["num_workers"] > 0 and device.type == 'cuda'))
        logger.info(f"数据加载器创建完成。训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本。")
    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        traceback.print_exc()
        return

    # --- 7. 初始化模型 ---
    logger.info("初始化模型...")
    model_config = config["model"]
    fe_config = config["feature_extractor"]
    feature_extractor: Optional[FeatureExtractorMTL] = None  # Initialize as None

    # a) 主 OCR 模型
    try:
        ocr_model = OCRModel(
            vocab_size=vocab_size, d_model=model_config["d_model"],
            decoder_nhead=model_config["decoder_nhead"], decoder_layers=model_config["decoder_layers"],
            decoder_dim_feedforward=model_config["decoder_dim_feedforward"],
            decoder_dropout=model_config["decoder_dropout"], max_seq_len=data_config["max_seq_len"],
            pad_token_id=pad_token_id, sos_token_id=sos_token_id, eos_token_id=eos_token_id,
            vit_model_name=model_config["vit_model_name"], vit_pretrained=model_config["vit_pretrained"]
        ).to(device)
        logger.info(f"OCR 模型初始化完成，总参数量: {sum(p.numel() for p in ocr_model.parameters()):,}")
        ocr_checkpoint_path = model_config.get("ocr_checkpoint_path")
        if ocr_checkpoint_path and ocr_checkpoint_path != "null":
            ocr_checkpoint_abs = os.path.abspath(
                os.path.join(project_root, ocr_checkpoint_path.lstrip('./').lstrip('../')))
            logger.info(f"尝试加载 OCR 模型初始权重从: {ocr_checkpoint_abs}")
            load_checkpoint(ocr_checkpoint_abs, ocr_model, map_location=device)
    except Exception as e:
        logger.error(f"OCR 模型初始化或加载权重失败: {e}")
        traceback.print_exc()
        return

    # b) 仅在 lambda_rl > 0 时加载预训练特征提取器
    if lambda_rl > 0:
        logger.info(f"RL 训练已启用 (lambda_rl={lambda_rl})，加载 Feature Extractor...")
        try:
            # Need to check if fe_config exists
            if not fe_config:
                logger.error("错误: RL 训练需要 feature_extractor 配置，但在 YAML 文件中未找到。")
                return

            feature_extractor = FeatureExtractorMTL(
                vocab_size=vocab_size, d_model=fe_config.get("d_model", 768), nhead=fe_config.get("nhead", 12),
                num_encoder_layers=fe_config.get("num_encoder_layers", 12),
                dim_feedforward=fe_config.get("dim_feedforward", 3072),
                dropout=fe_config.get("dropout", 0.1), max_seq_len=fe_config.get("max_seq_len", 256),
                pad_token_id=pad_token_id, max_bracket_depth=fe_config.get("max_bracket_depth", 10)
                # Use .get with defaults
            ).to(device)

            fe_checkpoint_path = fe_config.get("checkpoint_path")
            if not fe_checkpoint_path or fe_checkpoint_path == "null":
                logger.error(
                    "错误：RL 训练需要 Feature Extractor 检查点路径，但在配置中未指定 (feature_extractor.checkpoint_path)。")
                return
            fe_checkpoint_abs = os.path.abspath(
                os.path.join(project_root, fe_checkpoint_path.lstrip('./').lstrip('../')))
            logger.info(f"加载预训练 Feature Extractor 权重从: {fe_checkpoint_abs}")
            # Pass the model instance to load_checkpoint
            fe_checkpoint = load_checkpoint(fe_checkpoint_abs, feature_extractor, map_location=device)

            feature_extractor.eval()
            for param in feature_extractor.parameters():
                param.requires_grad = False
            logger.info("Feature Extractor 设置为评估模式，参数已冻结。")

        except Exception as e:
            logger.error(f"Feature Extractor 初始化或加载权重失败: {e}")
            traceback.print_exc()
            return
    else:
        logger.info(f"RL 训练已禁用 (lambda_rl={lambda_rl})，跳过加载 Feature Extractor。")

    # --- 8. 优化器和调度器 ---
    train_config = config["training"]
    try:
        optimizer = AdamW(filter(lambda p: p.requires_grad, ocr_model.parameters()),
                          lr=float(train_config["lr"]),
                          weight_decay=float(train_config["weight_decay"]))
        logger.info(
            f"优化器: AdamW (lr={train_config['lr']}, weight_decay={train_config['weight_decay']}) for OCR Model")
    except Exception as e:
        logger.error(f"创建优化器失败: {e}")
        traceback.print_exc()
        return
    num_epochs = train_config["epochs"]
    grad_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    if len(train_dataloader) == 0: logger.error("错误：训练数据加载器为空..."); return
    num_update_steps_per_epoch = max(1, len(train_dataloader) // grad_accumulation_steps)
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = train_config["warmup_steps"]
    if num_warmup_steps >= num_training_steps > 0:
        logger.warning(
            f"总训练步数 ({num_training_steps}) <= 预热步数 ({num_warmup_steps})..."); num_warmup_steps = max(0,
                                                                                                              num_training_steps - 1)
    elif num_training_steps <= 0:
        logger.error(f"计算得到的总训练步数 ({num_training_steps}) <= 0..."); return
    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "None")
    main_scheduler = None
    if scheduler_type == "CosineAnnealingLR":
        T_max = int((num_training_steps - num_warmup_steps) * scheduler_config.get("T_max_factor", 1.0));
        eta_min = float(scheduler_config.get("eta_min", 0))
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, T_max), eta_min=eta_min)
        logger.info(
            f"调度器: CosineAnnealingLR (T_max={max(1, T_max)}, eta_min={eta_min}) with Linear Warmup ({num_warmup_steps} steps)")
    elif scheduler_type == "LinearWarmupDecay":
        main_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        logger.info(f"调度器: LinearWarmupDecay (Warmup={num_warmup_steps}, Total={num_training_steps})")
    else:
        logger.info(f"调度器类型: {scheduler_type}. 将不使用主调度器。")

    # --- 9. 损失函数 ---
    criterion_ce = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    logger.info(f"损失函数: CrossEntropy (ignore_index={pad_token_id})"
                f"{f', RL Weight (lambda_rl)={lambda_rl}' if lambda_rl > 0 else ', RL 已禁用'}")

    # --- 10. 混合精度 ---
    use_amp = train_config["use_amp"]
    amp_dtype = None
    scaler_enabled = (use_amp and device.type == 'cuda')
    if use_amp:
        if device.type == 'cuda':
            amp_dtype = torch.float16
        elif device.type == 'cpu':
            amp_dtype = torch.bfloat16; logger.warning("AMP with bfloat16 on CPU is experimental.")
        else:
            use_amp = False; logger.warning(f"AMP is not supported on device type {device.type}. Disabling AMP.")
    scaler = GradScaler(enabled=(use_amp and scaler_enabled))
    logger.info(
        f"混合精度 (AMP): {'启用' if use_amp else '禁用'}, dtype: {amp_dtype}, GradScaler: {'启用' if scaler.is_enabled() else '禁用'}")

    # --- 11. 训练循环 ---
    logger.info("=" * 40)
    logger.info("开始训练循环...")
    logger.info(f"总轮数: {num_epochs}, 每轮更新次数: {num_update_steps_per_epoch}, 总步数: {num_training_steps}")
    logger.info("=" * 40)

    global_step = 0
    best_val_metric = -float('inf')
    start_epoch = 0

    # Optional: Resume training state
    resume_path = train_config.get("resume_checkpoint_path")
    if resume_path and resume_path != "null":
        resume_abs_path = os.path.abspath(os.path.join(project_root, resume_path.lstrip('./').lstrip('../')))
        if os.path.isfile(resume_abs_path):
            logger.info(f"尝试从检查点恢复训练: {resume_abs_path}")
            try:
                logger.info(f"调用 load_checkpoint 前，optimizer is None: {optimizer is None}")
                if optimizer is not None:
                    logger.info(f"调用 load_checkpoint 前，optimizer 类型: {type(optimizer)}")
                else:
                    logger.warning("!! 调用 load_checkpoint 前，optimizer 变量是 None !!")
                # Pass scheduler if needed for resumption (currently not implemented in load_checkpoint)
                checkpoint = load_checkpoint(resume_abs_path, ocr_model, optimizer=optimizer, scheduler=None,
                                             map_location=device)
                start_epoch = checkpoint.get('epoch', 0)
                best_val_metric = checkpoint.get('best_val_metric', -float('inf'))
                global_step = start_epoch * num_update_steps_per_epoch  # Recalculate step
                # --- Restore scheduler state if loaded ---
                # if main_scheduler and 'scheduler_state_dict' in checkpoint:
                #     try:
                #         main_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                #         logger.info("Scheduler state restored.")
                #     except Exception as e:
                #         logger.error(f"Error restoring scheduler state: {e}")
                # ------------------------------------------
                logger.info(f"从 Epoch {start_epoch} 恢复训练, 当前最佳指标: {best_val_metric:.4f}")
            except Exception as e:
                logger.error(f"无法从检查点 {resume_abs_path} 恢复: {e}. 将从头开始训练。")
                start_epoch = 0;
                global_step = 0;
                best_val_metric = -float('inf')
        else:
            logger.warning(f"指定的恢复检查点路径不存在: {resume_abs_path}. 将从头开始训练。")

    # --- Helper function for preparing sequences for Feature Extractor ---
    # Define it here, it will only be called if lambda_rl > 0
    def prepare_for_feature_extractor(sequences_ids: torch.Tensor, max_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Adds CLS, SEP, pads, and creates attention mask."""
        batch_size = sequences_ids.size(0)
        processed_ids = []
        attention_masks = []
        # Ensure CLS/SEP IDs are available before using them
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("CLS or SEP token ID is None, cannot prepare for feature extractor.")

        for i in range(batch_size):
            seq = sequences_ids[i].cpu().tolist()
            seq_clean = [tid for tid in seq if tid not in [pad_token_id, sos_token_id, eos_token_id]]
            input_seq = [cls_token_id] + seq_clean[:max_len - 2] + [sep_token_id]
            seq_len = len(input_seq)
            padding_len = max_len - seq_len
            input_ids = input_seq + [pad_token_id] * padding_len
            attn_mask = [1] * seq_len + [0] * padding_len
            processed_ids.append(input_ids)
            attention_masks.append(attn_mask)
        return torch.tensor(processed_ids, dtype=torch.long, device=device), \
            torch.tensor(attention_masks, dtype=torch.long, device=device)

    optimizer.zero_grad()  # Ensure gradients are zero before starting loop

    # --- Training Loop ---
    for epoch in range(start_epoch, num_epochs):
        ocr_model.train()

        epoch_loss_meter = AverageMeter()
        ce_loss_meter = AverageMeter()
        # Conditionally initialize RL meters
        rl_loss_meter = AverageMeter() if lambda_rl > 0 else None
        reward_meter = AverageMeter() if lambda_rl > 0 else None
        batch_time_meter = AverageMeter()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        for i, batch in pbar:
            if not batch: continue

            images = batch["image"].to(device, non_blocking=True)
            caption_ids = batch["caption_ids"].to(device, non_blocking=True)
            captions_in = caption_ids[:, :-1]
            captions_out = caption_ids[:, 1:]
            current_batch_size = images.size(0)

            # --- 1. Supervised Learning (Cross-Entropy) ---
            with autocast(enabled=use_amp, dtype=amp_dtype):
                logits = ocr_model(images, captions_in)
                ce_loss = criterion_ce(logits.reshape(-1, vocab_size), captions_out.reshape(-1))

            # --- 2. Reinforcement Learning (Conditional) ---
            rl_loss = torch.tensor(0.0, device=device)  # Initialize RL loss to 0
            final_reward_mean_for_log = 0.0  # Initialize reward for logging

            if lambda_rl > 0 and feature_extractor is not None:
                try:
                    # a) Sample from current policy
                    with torch.no_grad():
                        ocr_model.eval()
                        sampled_ids, log_probs_total = ocr_model.sample(images, max_len=data_config["max_seq_len"])
                        ocr_model.train()

                    # b) Calculate Base Reward (Cosine Similarity)
                    with torch.no_grad():
                        with autocast(enabled=use_amp, dtype=amp_dtype):
                            fe_max_len = fe_config.get("max_seq_len", 256)  # Use .get with default
                            sampled_ids_fe, sampled_mask_fe = prepare_for_feature_extractor(sampled_ids, fe_max_len)
                            gt_ids_fe, gt_mask_fe = prepare_for_feature_extractor(caption_ids, fe_max_len)

                            pooling_strategy = fe_config.get('pooling_strategy', 'mean')
                            feat_gen = feature_extractor.encode(sampled_ids_fe, sampled_mask_fe,
                                                                pooling_strategy=pooling_strategy)
                            feat_real = feature_extractor.encode(gt_ids_fe, gt_mask_fe,
                                                                 pooling_strategy=pooling_strategy)

                            base_reward = F.cosine_similarity(feat_gen, feat_real, dim=-1)

                    # c) Calculate Exact Match Bonus Reward
                    exact_match_bonus_value = config["rl"].get("exact_match_bonus", 0.0)
                    bonus_reward = torch.zeros_like(base_reward)

                    if exact_match_bonus_value > 0:
                        for k in range(current_batch_size):
                            gen_seq_tensor = sampled_ids[k].cpu()
                            gt_seq_tensor = caption_ids[k].cpu()
                            # (Comparison logic - ensure SOS/EOS/PAD handling is correct)
                            gen_eos_indices = (gen_seq_tensor == eos_token_id).nonzero(as_tuple=True)[0]
                            gen_pad_indices = (gen_seq_tensor == pad_token_id).nonzero(as_tuple=True)[0]
                            gen_end_idx = gen_seq_tensor.size(0)
                            if len(gen_eos_indices) > 0: gen_end_idx = min(gen_end_idx, gen_eos_indices[0].item())
                            if len(gen_pad_indices) > 0: gen_end_idx = min(gen_end_idx, gen_pad_indices[0].item())
                            # Start comparison after SOS token (index 1)
                            gen_ids_effective = gen_seq_tensor[1:gen_end_idx].tolist()

                            gt_eos_indices = (gt_seq_tensor == eos_token_id).nonzero(as_tuple=True)[0]
                            gt_pad_indices = (gt_seq_tensor == pad_token_id).nonzero(as_tuple=True)[0]
                            gt_end_idx = gt_seq_tensor.size(0)
                            if len(gt_eos_indices) > 0: gt_end_idx = min(gt_end_idx, gt_eos_indices[0].item())
                            if len(gt_pad_indices) > 0: gt_end_idx = min(gt_end_idx, gt_pad_indices[0].item())
                            # Start comparison after SOS token (index 1)
                            gt_ids_effective = gt_seq_tensor[1:gt_end_idx].tolist()

                            if gen_ids_effective == gt_ids_effective:
                                bonus_reward[k] = exact_match_bonus_value

                    # d) Calculate Final Reward
                    # TODO：奖励策略需要优化
                    final_reward = base_reward + bonus_reward
                    final_reward_mean_for_log = final_reward.mean().item()  # For logging

                    # e) Calculate Baseline
                    baseline = final_reward.mean()

                    # f) Calculate RL Loss
                    advantage = final_reward - baseline
                    if log_probs_total is not None:
                        rl_loss = -(log_probs_total * advantage.detach()).mean()
                    else:
                        logger.warning("log_probs_total is None, skipping RL loss calculation for this batch.")
                        rl_loss = torch.tensor(0.0, device=device)

                except Exception as rl_error:  # Catch errors specifically in the RL block
                    logger.error(f"Error during RL calculation in batch {i}: {rl_error}", exc_info=True)
                    # Decide how to handle: skip batch update? Fallback to CE only?
                    # For now, let rl_loss remain 0 for this batch
                    rl_loss = torch.tensor(0.0, device=device)
                    final_reward_mean_for_log = 0.0  # Reset log value

            # --- 3. Combine Losses ---
            with autocast(enabled=use_amp, dtype=amp_dtype):
                total_loss = ce_loss + lambda_rl * rl_loss  # lambda_rl weights the entire RL contribution

            # --- 4. Backpropagation & Optimization ---
            # Normalize loss for accumulation *before* scaling
            total_loss_scaled_for_accum = total_loss / grad_accumulation_steps
            scaler.scale(total_loss_scaled_for_accum).backward()

            if (i + 1) % grad_accumulation_steps == 0:
                # Unscale before clipping (only if using scaler)
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)

                # Gradient Clipping (apply regardless of scaler, but after unscaling)
                max_grad_norm = train_config.get("max_grad_norm")
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, ocr_model.parameters()),
                                                   float(max_grad_norm))

                # Optimizer Step (scaler handles enabled/disabled state)
                scaler.step(optimizer)
                scaler.update()  # Update scaler state

                # Scheduler Step
                if main_scheduler is not None:
                    # Step based on global steps, respecting warmup
                    if scheduler_type == "LinearWarmupDecay":
                        main_scheduler.step()
                    elif global_step >= num_warmup_steps:
                        # Only step cosine/other schedulers after warmup
                        main_scheduler.step()

                optimizer.zero_grad()  # Reset gradients for next cycle
                global_step += 1  # Increment global step on optimizer step

            # --- 5. Logging ---
            batch_time_meter.update(time.time() - start_time)
            start_time = time.time()
            epoch_loss_meter.update(total_loss.item(), current_batch_size)
            ce_loss_meter.update(ce_loss.item(), current_batch_size)
            # Conditionally update RL meters
            if rl_loss_meter is not None:
                rl_loss_meter.update(rl_loss.item(), current_batch_size)
            if reward_meter is not None:
                reward_meter.update(final_reward_mean_for_log, current_batch_size)

            # --- Set postfix for progress bar ---
            postfix_dict = {
                "Loss": f"{epoch_loss_meter.avg:.4f}",
                "CE": f"{ce_loss_meter.avg:.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            }
            if lambda_rl > 0 and rl_loss_meter is not None and reward_meter is not None:
                postfix_dict["RL"] = f"{rl_loss_meter.avg:.4f}"
                postfix_dict["Reward"] = f"{reward_meter.avg:.3f}"  # Shows final reward avg
            pbar.set_postfix(postfix_dict)
            # --- End Batch Loop ---

        # --- End of Epoch ---
        log_msg = (f"Epoch {epoch + 1} 训练结束. "
                   f"Avg Loss: {epoch_loss_meter.avg:.4f}, "
                   f"Avg CE Loss: {ce_loss_meter.avg:.4f}")
        if lambda_rl > 0 and rl_loss_meter is not None and reward_meter is not None:
            log_msg += (f", Avg RL Loss: {rl_loss_meter.avg:.4f}, "
                        f"Avg Reward: {reward_meter.avg:.4f}")  # Log final reward avg
        log_msg += f", Time: {batch_time_meter.sum:.2f}s"
        logger.info(log_msg)

        # --- 12. 验证 ---
        # TODO：性能需要优化
        if (epoch + 1) % train_config["validation_interval"] == 0:
            logger.info(f"--- 开始验证 Epoch {epoch + 1} ---")
            ocr_model.eval()  # Ensure eval mode for validation
            all_references_ids = []
            all_hypotheses_ids = []
            eval_config = config["evaluation"]
            with torch.no_grad():
                val_pbar = tqdm(val_dataloader, desc="Validation")
                for batch in val_pbar:
                    if not batch: continue
                    images = batch["image"].to(device, non_blocking=True)
                    caption_ids_list = batch["caption_ids_list"]  # Assumes collate_fn provides this
                    try:
                        with autocast(enabled=use_amp, dtype=amp_dtype):  # Use AMP for generation if enabled
                            generated_ids = ocr_model.generate(
                                images,
                                max_len=data_config["max_seq_len"],
                                method=eval_config.get("generation_method", "greedy"),
                                beam_width=eval_config.get("beam_width", 5),
                                length_penalty=eval_config.get("length_penalty", 0.7)
                            ).cpu().tolist()
                        all_references_ids.extend(caption_ids_list)
                        all_hypotheses_ids.extend(generated_ids)
                    except Exception as gen_error:
                        logger.error(f"Error during generation in validation: {gen_error}", exc_info=True)
                        # Decide how to handle, e.g., skip batch?
                        continue  # Skip this batch if generation fails

            # Calculate evaluation metrics
            if not all_references_ids or not all_hypotheses_ids:
                logger.warning("验证期间未生成引用或假设，跳过指标计算。")
                metrics = {}
            else:
                if len(all_references_ids) != len(all_hypotheses_ids):
                    logger.error(
                        f"Validation Error: Refs ({len(all_references_ids)}) != Hyps ({len(all_hypotheses_ids)})!")
                    metrics = {}
                else:
                    try:
                        metrics = compute_metrics(
                            references_ids=all_references_ids,
                            hypotheses_ids=all_hypotheses_ids,
                            rev_vocab=rev_vocab,
                            vocab=vocab,
                            eos_token='<EOS>',
                            pad_token='<PAD>',
                            sos_token='<SOS>'
                        )
                        logger.info(f"验证结果 Epoch {epoch + 1}:")
                        for name, value in metrics.items(): logger.info(f"  {name}: {value:.4f}")
                    except Exception as e:
                        logger.error(f"计算指标时出错: {e}", exc_info=True)
                        metrics = {}

            # --- 13. 保存检查点 ---
            primary_metric = eval_config.get("primary_metric", "ExactMatch")
            current_metric = metrics.get(primary_metric, metrics.get("BLEU", -1.0))
            # Check if current_metric is valid (not NaN, etc.) before comparison
            if isinstance(current_metric, (int, float)) and np.isfinite(current_metric):
                is_best = current_metric > best_val_metric
                if is_best:
                    best_val_metric = current_metric
                    logger.info(f"*** 新的最佳验证指标 ({primary_metric}): {best_val_metric:.4f} ***")
            else:
                logger.warning(f"当前验证指标 '{primary_metric}' 无效 ({current_metric})，无法判断是否最佳。")
                is_best = False  # Treat invalid metric as not best

            checkpoint_state = {
                'epoch': epoch + 1,
                'ocr_model_state_dict': ocr_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_metric': best_val_metric,
                # --- Save scheduler state ---
                'scheduler_state_dict': main_scheduler.state_dict() if main_scheduler else None,
                # --- Save scaler state ---
                'scaler_state_dict': scaler.state_dict() if scaler.is_enabled() else None,
                'config': config
            }
            save_interval = train_config.get("save_every_n_epochs", 0)
            save_epoch_checkpoint = (save_interval > 0 and (epoch + 1) % save_interval == 0)
            epoch_filename = f'checkpoint_epoch_{epoch + 1}.pth.tar' if save_epoch_checkpoint else None
            best_filename = 'model_best.pth.tar'
            save_checkpoint(state=checkpoint_state, is_best=is_best, checkpoint_dir=output_dir, filename=epoch_filename,
                            best_filename=best_filename)
            save_actions = []
            if is_best: save_actions.append(f"最佳模型已更新 ({primary_metric}: {best_val_metric:.4f})")
            if epoch_filename: save_actions.append(f"Epoch {epoch + 1} 检查点已保存")
            if save_actions: logger.info(f"检查点操作: {', '.join(save_actions)} 到 {output_dir}")

    # --- End Training Loop ---
    logger.info("=" * 40)
    logger.info("训练结束。")
    logger.info(f"最终最佳验证指标 ({primary_metric}): {best_val_metric:.4f}")
    logger.info(f"最佳模型保存在: {os.path.join(output_dir, best_filename)}")
    logger.info("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train OCR Model with optional Reinforcement Learning")
    default_config_path = os.path.join(project_root, "configs", "train_rl_config.yaml")
    parser.add_argument('--config', type=str, default=default_config_path,
                        help='Path to the training configuration YAML file')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"错误: 配置文件未找到于 {args.config}")
        script_dir = os.path.dirname(current_script_path)
        fallback_config_path = os.path.join(script_dir, os.path.basename(args.config))
        if os.path.exists(fallback_config_path):
            print(f"尝试使用相对路径: {fallback_config_path}"); args.config = fallback_config_path
        else:
            fallback_config_path_2 = os.path.join(os.path.dirname(script_dir), "configs", os.path.basename(args.config))
            if os.path.exists(fallback_config_path_2):
                print(f"尝试使用 ../configs/ 路径: {fallback_config_path_2}"); args.config = fallback_config_path_2
            else:
                sys.exit(1)
    train_rl(args.config)
