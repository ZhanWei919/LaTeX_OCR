import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.cuda.amp import GradScaler, autocast
import json
import os
import time
from tqdm import tqdm
import logging
import yaml
import argparse
import random
import numpy as np


import sys
import traceback


current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:  # 避免重复添加
    sys.path.append(project_root)
    print(f"添加到 sys.path: {project_root}")

from models.feature_extractor import FeatureExtractorMTL
from models.dataloader_mtl import MTLDataset, tokenize_latex, collate_mtl
from models.utils import AverageMeter  # Assuming utils.py exists


# --- Logger Setup ---
def setup_logger(log_file):
    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir:  # Only create if dirname is not empty
        os.makedirs(log_dir, exist_ok=True)
    # Remove existing handlers to avoid duplicate logs if called multiple times
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),  # 指定编码
            logging.StreamHandler(sys.stdout)  # 同时输出到控制台
        ]
    )
    return logging.getLogger(__name__)  # Use specific logger name


# --- Learning Rate Warmup ---
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Linear warmup and decay scheduler. """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# --- Main Pre-training Function ---
def pretrain(config_path: str):
    # --- Load Configuration ---
    try:
        with open(config_path, 'r', encoding='utf-8') as f:  # 指定编码
            config = yaml.safe_load(f)
        print("配置加载成功。")
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        traceback.print_exc()
        return

    # --- Setup Output Dir and Logger ---
    # 使用绝对路径或相对于项目根目录的路径更可靠
    output_dir_rel = config["training"]["output_dir"]
    log_file_rel = config["training"]["log_file"]
    output_dir = os.path.abspath(os.path.join(project_root, output_dir_rel.lstrip('./').lstrip('../')))
    log_file = os.path.abspath(os.path.join(project_root, log_file_rel.lstrip('./').lstrip('../')))

    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger(log_file)  # 现在日志会写入文件并打印到控制台
    logger.info(f"配置从 {config_path} 加载。")
    logger.info(f"输出将保存到: {output_dir}")
    logger.info(f"日志将记录到: {log_file}")

    # --- Set Seed ---
    seed = config["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logger.info("CUDA 可用，已设置 CUDA 随机种子。")
    else:
        logger.info("CUDA 不可用，仅设置 CPU 随机种子。")
    logger.info(f"全局随机种子设置为: {seed}")

    # --- Device ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"检测到 CUDA 设备: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("未检测到 CUDA 设备，将使用 CPU。")
    logger.info(f"使用设备: {device}")

    # --- Load Vocab ---
    vocab_file_rel = config["data"]["vocab_file"]
    vocab_file_abs = os.path.abspath(os.path.join(project_root, vocab_file_rel.lstrip('./').lstrip('../')))
    try:
        with open(vocab_file_abs, 'r', encoding='utf-8') as f:  # 指定编码
            vocab = json.load(f)
        vocab_size = len(vocab)
        pad_token_id = vocab.get('<PAD>', 0)
        logger.info(f"词汇表加载: {vocab_size} tokens. PAD ID: {pad_token_id}")
    except Exception as e:
        logger.error(f"无法加载词汇表 {vocab_file_abs}: {e}")
        traceback.print_exc()
        return

    # --- Create Dataset and DataLoader ---
    logger.info("创建数据加载器...")
    try:
        json_file_path_rel = config["data"]["train_split_file"]  # 使用 YAML 中的键名
        json_file_abs = json_file_path_rel  # 先假设是绝对路径
        if not os.path.isabs(json_file_path_rel):  # 如果不是绝对路径，则拼接
            json_file_abs = os.path.abspath(os.path.join(project_root, json_file_path_rel.lstrip('./').lstrip('../')))

        logger.info(f"将从 JSON 文件加载数据: {json_file_abs}")
        if not os.path.exists(json_file_abs):
            logger.error(f"错误: JSON 文件不存在于路径: {json_file_abs}")
            return

        dataset = MTLDataset(
            json_file=json_file_abs,  # 传入 JSON 文件路径
            vocab=vocab,
            tokenizer=tokenize_latex,
            max_seq_len=config["data"]["max_seq_len"],
            ignore_index=-100,
            max_bracket_depth=config["model"]["max_bracket_depth"],
            data_fraction=config["data"]["data_fraction"]
        )

        if len(dataset) == 0:
            logger.error("错误：数据集中没有加载到任何样本，请检查 JSON 文件和路径。")
            return

        dataloader = DataLoader(
            dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            # 在 Windows CPU 上 num_workers > 0 可能导致问题，可以先设为 0 测试
            num_workers=config["data"]["num_workers"] if device.type == 'cuda' else 0,
            collate_fn=collate_mtl,
            pin_memory=(device.type == 'cuda'),  # pin_memory 只在 CUDA 上有效
            persistent_workers=(config["data"]["num_workers"] > 0 and device.type == 'cuda')
        )
        logger.info(f"数据加载器创建完成，共 {len(dataset)} 样本。")
        if device.type == 'cpu' and config["data"]["num_workers"] > 0:
            logger.warning("在 CPU 模式下，建议将 num_workers 设置为 0 以避免潜在问题。当前 num_workers > 0。")

    except Exception as e:
        logger.error(f"创建数据加载器失败: {e}")
        traceback.print_exc()
        return

    # --- Initialize Model ---
    logger.info("初始化模型...")
    model_config = config["model"]
    try:
        model = FeatureExtractorMTL(
            vocab_size=vocab_size,
            d_model=model_config["d_model"],
            nhead=model_config["nhead"],
            num_encoder_layers=model_config["num_encoder_layers"],
            dim_feedforward=model_config["dim_feedforward"],
            dropout=model_config["dropout"],
            max_seq_len=config["data"]["max_seq_len"],
            pad_token_id=pad_token_id,
            max_bracket_depth=model_config["max_bracket_depth"]
        ).to(device)
        logger.info(f"模型初始化完成，总参数量: {sum(p.numel() for p in model.parameters()):,}")
    except Exception as e:
        logger.error(f"模型初始化失败: {e}")
        traceback.print_exc()
        return

    # --- Optimizer ---
    train_config = config["training"]
    # --- Explicitly convert lr and weight_decay to float ---
    try:
        learning_rate = float(train_config["lr"])
        weight_decay_val = float(train_config["weight_decay"])
    except ValueError as e:
        logger.error(f"无法将配置中的 lr 或 weight_decay 转换为 float: {e}")
        logger.error(f"LR value: {train_config.get('lr')}, Weight Decay value: {train_config.get('weight_decay')}")
        traceback.print_exc()
        return  # Exit if conversion fails
    except KeyError as e:
        logger.error(f"配置中缺少键: {e}")
        traceback.print_exc()
        return  # Exit if key is missing

    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,  # Use the converted float value
                      weight_decay=weight_decay_val)  # Use the converted float value
    logger.info(f"优化器: AdamW (lr={learning_rate}, weight_decay={weight_decay_val})")  # Log the converted values

    # --- Scheduler ---
    num_epochs = train_config["epochs"]
    grad_accumulation_steps = train_config.get("gradient_accumulation_steps", 1)
    if len(dataloader) == 0:
        logger.error("Dataloader 为空，无法计算训练步数。")
        return
    num_update_steps_per_epoch = max(1, len(dataloader) // grad_accumulation_steps)  # 确保至少为 1
    if len(dataloader) > 0 and len(dataloader) < grad_accumulation_steps:
        logger.warning(
            f"Dataloader size ({len(dataloader)}) is smaller than gradient_accumulation_steps ({grad_accumulation_steps}). 将在每个 epoch 结束时执行一次更新。")
        num_update_steps_per_epoch = 1  # 确保至少更新一次

    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = train_config["warmup_steps"]

    if num_training_steps <= num_warmup_steps and num_training_steps > 0:
        logger.warning(
            f"Total training steps ({num_training_steps}) is less than or equal to warmup steps ({num_warmup_steps}). 预热步数将调整为 {max(0, num_training_steps - 1)}.")
        num_warmup_steps = max(0, num_training_steps - 1)
    elif num_training_steps == 0:
        logger.error(
            "计算得到的总训练步数为 0，无法继续训练。请检查 epoch, batch_size, gradient_accumulation_steps 和数据量。")
        return

    scheduler_config = config.get("scheduler", {})
    scheduler_type = scheduler_config.get("type", "None")
    main_scheduler = None

    if scheduler_type == "CosineAnnealingLR":
        T_max = int((num_training_steps - num_warmup_steps) * scheduler_config.get("T_max_factor", 1.0))
        eta_min = float(scheduler_config.get("eta_min", 0))  # 确保 eta_min 是 float
        main_scheduler = CosineAnnealingLR(optimizer, T_max=max(1, T_max), eta_min=eta_min)
        logger.info(
            f"调度器: CosineAnnealingLR (T_max={max(1, T_max)}, eta_min={eta_min}) with Linear Warmup ({num_warmup_steps} steps)")
    elif scheduler_type == "LinearWarmupDecay":
        main_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
        logger.info(f"调度器: LinearWarmupDecay (Warmup={num_warmup_steps}, Total={num_training_steps})")
    else:
        logger.info(f"调度器类型: {scheduler_type}. 将不使用主调度器 (只有预热)。")

    # --- Loss Functions ---
    criterion_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    criterion_structure = nn.MSELoss(reduction='none')  # MSE 先不 reduce，手动处理 mask
    loss_weights = config["loss_weights"]
    logger.info(f"损失权重: MLM={loss_weights['mlm']}, Structure={loss_weights['structure']}")

    # --- Mixed Precision Scaler & Dtype ---
    use_amp = train_config["use_amp"]
    amp_dtype = None  # 初始化 amp 数据类型为 None

    # GradScaler 仅在 CUDA 上且 use_amp 为 True 时启用
    scaler_enabled = (use_amp and device.type == 'cuda')
    scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)

    logger.info(f"配置请求使用混合精度 (AMP): {use_amp}")

    if use_amp:
        if device.type == 'cuda':
            amp_dtype = torch.float16
            logger.info(f"CUDA AMP 已启用, 使用 dtype: {amp_dtype}, GradScaler: 已启用")
        elif device.type == 'cpu':
            # CPU AMP 尝试使用 bfloat16
            try:
                # 检查 bfloat16 是否真的可用 (某些 PyTorch 版本或 CPU 可能形式上支持但计算有问题)
                # 这里简单地假设如果 torch.bfloat16 存在即可用
                amp_dtype = torch.bfloat16
                logger.info(f"CPU AMP 已启用 (实验性), 使用 dtype: {amp_dtype}, GradScaler: 已禁用")
                # 确保 scaler 仍是禁用的 (虽然上面已处理，双重保险)
                scaler = torch.amp.GradScaler(device.type, enabled=scaler_enabled)
            except AttributeError:
                logger.warning("当前 PyTorch 版本或 CPU 不支持 torch.bfloat16, CPU AMP 将被禁用。")
                use_amp = False  # 禁用 AMP 如果 bfloat16 不可用
                amp_dtype = None
                scaler = GradScaler(enabled=False)  # 明确禁用 scaler
        else:
            logger.warning(f"未知的设备类型 '{device.type}', AMP 将被禁用。")
            use_amp = False  # 禁用 AMP 对于未知设备
            amp_dtype = None  # 确保 amp_dtype 为 None
            scaler = GradScaler(enabled=False)  # 明确禁用 scaler
    else:
        logger.info("AMP 已禁用。")
        scaler = GradScaler(enabled=False)  # 明确禁用 scaler

    # --- Training Loop ---
    logger.info("=" * 40)
    logger.info("开始预训练循环...")
    logger.info(f"总训练轮数: {num_epochs}")
    logger.info(f"每轮步数 (数据加载次数): {len(dataloader)}")
    logger.info(f"梯度累积步数: {grad_accumulation_steps}")
    logger.info(f"每轮更新次数 (优化器步数): {num_update_steps_per_epoch}")
    logger.info(f"总训练步数 (优化器步数): {num_training_steps}")
    logger.info("=" * 40)

    global_step = 0
    optimizer.zero_grad()  # Ensure grads are zero at the start

    for epoch in range(num_epochs):
        model.train()
        epoch_loss_meter = AverageMeter()
        mlm_loss_meter = AverageMeter()
        structure_loss_meter = AverageMeter()
        batch_time_meter = AverageMeter()
        data_time_meter = AverageMeter()

        pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        start_time = time.time()

        for i, batch in pbar:
            # Handle potential empty batch from collate_fn if dataloader was empty
            if not batch:
                logger.warning(f"跳过空批次，索引 {i}。")
                continue

            data_time_meter.update(time.time() - start_time)

            # Move batch to device
            try:
                input_ids = batch["input_ids"].to(device, non_blocking=(device.type == 'cuda'))
                attention_mask = batch["attention_mask"].to(device, non_blocking=(device.type == 'cuda'))
                mlm_labels = batch["mlm_labels"].to(device, non_blocking=(device.type == 'cuda'))
                structure_labels = batch["structure_labels"].to(device, non_blocking=(device.type == 'cuda')).float()
            except KeyError as e:
                logger.error(f"批次数据缺少键: {e}。跳过批次。")
                continue
            except Exception as e:
                logger.error(f"移动批次到设备时出错: {e}。跳过批次。")
                traceback.print_exc()
                continue

            # --- Forward pass with autocast ---
            # 使用 device-agnostic 的 torch.autocast
            # enabled=use_amp 控制是否实际启用，amp_dtype 指定类型 (如果启用)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                try:
                    mlm_logits, structure_logits = model(input_ids, attention_mask)
                    # mlm_logits: [B, seq_len, vocab_size]
                    # structure_logits: [B, seq_len, 1]

                    # --- Calculate MLM Loss ---
                    # .view(-1, N) is generally safe
                    loss_mlm = criterion_mlm(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))

                    # --- Calculate Structure Loss (Masked MSE) ---
                    structure_preds = structure_logits.squeeze(-1)  # [B, seq_len]
                    # Ensure structure_labels is also float for MSE
                    loss_struct_raw = criterion_structure(structure_preds, structure_labels.float())  # [B, seq_len]
                    # Ensure mask is float for multiplication
                    mask = attention_mask.float()  # [B, seq_len]
                    loss_struct_masked = loss_struct_raw * mask
                    # Calculate mean loss only over non-masked elements
                    mask_sum = mask.sum()
                    if mask_sum > 0:
                        loss_structure = loss_struct_masked.sum() / mask_sum
                    else:
                        # Handle case where mask is all zeros (e.g., empty batch or all padding)
                        loss_structure = torch.tensor(0.0, device=device)

                    # --- Combine Losses ---
                    # Ensure loss_weights are float
                    w_mlm = float(loss_weights["mlm"])
                    w_struct = float(loss_weights["structure"])
                    total_loss = w_mlm * loss_mlm + w_struct * loss_structure

                    # Check for NaN loss
                    if torch.isnan(total_loss):
                        logger.warning(
                            f"检测到 NaN 损失！ MLM Loss: {loss_mlm.item()}, Structure Loss: {loss_structure.item()}. 跳过此批次更新。")
                        # 重置梯度并跳过后续步骤
                        optimizer.zero_grad()
                        continue

                    # Normalize loss for gradient accumulation
                    total_loss = total_loss / grad_accumulation_steps

                except Exception as e:
                    logger.error(f"前向传播或损失计算期间出错: {e}")
                    traceback.print_exc()
                    continue  # Skip this batch

            # --- Backward Pass ---
            # scaler.scale(total_loss).backward()
            # scaler is a no-op if disabled
            try:
                scaler.scale(total_loss).backward()
            except Exception as e:
                logger.error(f"前向传播或损失计算期间出错: {e}")
                traceback.print_exc()
                # Optionally clear gradients and skip optimizer step
                optimizer.zero_grad()
                continue  # Skip optimizer step for this batch

            # --- Optimizer Step (after accumulation) ---
            if (i + 1) % grad_accumulation_steps == 0:
                # Optional: Gradient Clipping (before optimizer step)
                max_grad_norm = train_config.get("max_grad_norm")  # 获取梯度裁剪值
                if max_grad_norm is not None:
                    # Unscale gradients before clipping, only if scaler is enabled
                    if scaler.is_enabled():  # Check if scaler is actually enabled
                        scaler.unscale_(optimizer)
                    # Clip gradients regardless of scaler, but after unscaling if applicable
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   float(max_grad_norm))  # Ensure max_grad_norm is float

                # scaler.step() is a no-op if scaler is disabled
                scaler.step(optimizer)
                # scaler.update() is a no-op if scaler is disabled
                scaler.update()

                # --- Learning Rate Warmup/Step ---
                current_lr = optimizer.param_groups[0]['lr']  # Get current LR for logging
                # Warmup logic (applied before main scheduler step)
                if global_step < num_warmup_steps:
                    # Calculate warmup LR based on the *initial* learning rate
                    lr_scale = float(global_step + 1) / float(max(1, num_warmup_steps))
                    for group in optimizer.param_groups:
                        group['lr'] = learning_rate * lr_scale  # Use initial LR from config
                elif main_scheduler is not None:  # Step main scheduler *after* warmup is complete
                    # Check if scheduler step should happen based on global_step or epoch
                    main_scheduler.step()  # Step based on optimizer steps

                optimizer.zero_grad()  # Reset gradients for the next accumulation cycle
                global_step += 1  # Increment global step only on optimizer step

            # --- Logging ---
            batch_size = input_ids.size(0)
            # Use .item() safely, handle potential non-tensor loss (though unlikely now)
            current_total_loss = total_loss.item() * grad_accumulation_steps if torch.is_tensor(
                total_loss) else total_loss * grad_accumulation_steps
            current_mlm_loss = loss_mlm.item() if torch.is_tensor(loss_mlm) else loss_mlm
            current_struct_loss = loss_structure.item() if torch.is_tensor(loss_structure) else loss_structure

            epoch_loss_meter.update(current_total_loss, batch_size)
            mlm_loss_meter.update(current_mlm_loss, batch_size)
            structure_loss_meter.update(current_struct_loss, batch_size)
            batch_time_meter.update(time.time() - start_time)
            start_time = time.time()  # Reset start time for next batch

            # Update tqdm postfix using current optimizer LR
            current_optimizer_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                "Loss": f"{epoch_loss_meter.avg:.4f}",
                "MLM": f"{mlm_loss_meter.avg:.4f}",
                "Struct": f"{structure_loss_meter.avg:.4f}",
                "LR": f"{current_optimizer_lr:.2e}"  # Use actual current LR
            })

        # --- End of Epoch ---
        logger.info(f"Epoch {epoch + 1} finished. "
                    f"Avg Loss: {epoch_loss_meter.avg:.4f}, "
                    f"Avg MLM Loss: {mlm_loss_meter.avg:.4f}, "
                    f"Avg Structure Loss: {structure_loss_meter.avg:.4f}, "
                    f"Time: {batch_time_meter.sum:.2f}s")

        # --- Save Checkpoint ---
        checkpoint_path = os.path.join(output_dir, f"epoch_{epoch + 1}.pt")
        try:
            # 保存模型状态、优化器状态、调度器状态和配置
            save_dict = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }
            if main_scheduler:
                save_dict['scheduler_state_dict'] = main_scheduler.state_dict()
            if scaler.is_enabled():
                save_dict['scaler_state_dict'] = scaler.state_dict()

            torch.save(save_dict, checkpoint_path)
            logger.info(f"检查点已保存到 {checkpoint_path}")

        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
            traceback.print_exc()

    logger.info("=" * 40)
    logger.info("预训练结束。")
    logger.info("=" * 40)


if __name__ == "__main__":
    config_file = "pretrain_config.yaml"
    config_path = os.path.join(project_root, "configs", config_file)

    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件未找到于 {config_path}")
    else:
        pretrain(config_path)
