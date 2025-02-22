import time
import config
import torch.optim
import torch.utils.data
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from model.utils import *
from model import metrics, dataloader
from model.model import Model
from torch.utils.checkpoint import checkpoint as train_ck
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import logging
from datetime import datetime
import os
from torch.utils.data import DataLoader
from model.dataloader import collate_fn
import cv2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


save_freq = 5
Model.device = device

cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def setup_logger():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    return logging.getLogger()

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    global best_score, epochs_since_improvement, checkpoint, start_epoch
    logger = setup_logger()
    word_map = load_json(config.vocab_path)
    if config.checkpoint is None:
        model = Model(vocab_size=len(word_map), embed_dim=config.emb_dim, decoder_dim=config.decoder_dim, encoder_dim=512,
                      disc_hidden_dim=256, fusion_dim=config.fusion_dim, ast_vocab_size=config.ast_vocab_size, 
                      ast_embed_dim=config.ast_embed_dim, ast_hidden_dim=config.ast_hidden_dim)
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=config.encoder_lr)
        decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=config.decoder_lr)
        disc_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=config.decoder_lr)
        start_epoch = 0
        best_score = 0
        epochs_since_improvement = 0
    else:
        checkpoint = torch.load(config.checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_score = checkpoint['score']
        model = checkpoint['model']
        encoder_optimizer = checkpoint['encoder_optimizer']
        decoder_optimizer = checkpoint['decoder_optimizer']
        disc_optimizer = checkpoint['disc_optimizer']

    model = model.to(device)

    if config.use_compile:
        model = torch.compile(model, mode=config.compile_mode)

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = dataloader.formuladataset(config.train_set_path, batch_size=config.batch_size, ratio=5)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, collate_fn=collate_fn)
    val_dataset = dataloader.formuladataset(config.val_set_path, batch_size=config.batch_size, ratio=5)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory, collate_fn=collate_fn)
    scaler = GradScaler()

    
    for epoch in range(start_epoch, config.epochs):
        train(train_loader, model, criterion, encoder_optimizer, decoder_optimizer, disc_optimizer, epoch, scaler)
        score = validate(val_loader, model, criterion, logger)
        is_best = score > best_score
        best_score = max(score, best_score)
        if not is_best:
            epochs_since_improvement += 1
            if epochs_since_improvement >= 20:
                logger.info('No improvement in 20 epochs - stopping training')
                save_checkpoint({
                    'epoch': epoch,
                    'epochs_since_improvement': epochs_since_improvement,
                    'score': score,
                    'model': model,
                    'encoder_optimizer': encoder_optimizer,
                    'decoder_optimizer': decoder_optimizer,
                    'disc_optimizer': disc_optimizer
                }, is_best, final=True)
                break
        else:
            epochs_since_improvement = 0
        if epoch % config.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'epochs_since_improvement': epochs_since_improvement,
                'score': score,
                'model': model,
                'encoder_optimizer': encoder_optimizer,
                'decoder_optimizer': decoder_optimizer,
                'disc_optimizer': disc_optimizer
            }, is_best)

    if epoch == config.epochs - 1:
        logger.info('Reached maximum epochs - saving final model')
        save_checkpoint({
            'epoch': epoch,
            'epochs_since_improvement': epochs_since_improvement,
            'score': score,
            'model': model,
            'encoder_optimizer': encoder_optimizer,
            'decoder_optimizer': decoder_optimizer,
            'disc_optimizer': disc_optimizer
        }, is_best, final=True)

def check_memory_usage():
    memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    if memory_allocated > 20:
        print(f"警告: 显存使用达到 {memory_allocated:.1f}GB")
        torch.cuda.empty_cache()
        return True
    return False

def train(train_loader, model, criterion, encoderOptimizer, decoderOptimizer, discOptimizer, epoch, scaler):
    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    start = time.time()

    # 添加梯度累积
    accumulation_steps = 4  # 每4个小批次更新一次
    optimizer_step = 0

    for i, (imgs, caps, caplens, ast_nodes, ast_structures) in enumerate(train_loader):
        print(f"\nBatch {i}: Starting processing...")
        data_time.update(time.time() - start)
        check_memory_usage()  # 检查显存使用

        print("Sorting batch data...")
        # 按 caplens 降序排序批次数据
        sorted_indices = torch.argsort(caplens, descending=True)
        imgs = torch.index_select(imgs, 0, sorted_indices)
        caps = torch.index_select(caps, 0, sorted_indices)
        caplens = torch.index_select(caplens, 0, sorted_indices)
        ast_nodes = torch.index_select(ast_nodes, 0, sorted_indices)
        ast_structures = [ast_structures[i] for i in sorted_indices]

        print("Moving data to device...")
        # 移动到设备
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        ast_nodes = ast_nodes.to(device)

        # 只在累积开始时清零梯度
        if optimizer_step % accumulation_steps == 0:
            encoderOptimizer.zero_grad()
            decoderOptimizer.zero_grad()
            discOptimizer.zero_grad()

        with autocast(device_type="cuda"):
            print("Running encoder...")
            encoder_out = model.encoder(imgs)
            print("Running decoder...")
            scores, caps_sorted, decode_lengths = model.decoder(encoder_out, caps, caplens)
            targets = caps_sorted[:, 1:]
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss_content = criterion(scores_packed, targets_packed)

            print("Starting REINFORCE training...")
            # REINFORCE 训练
            gen_seqs, _ = model.generate(imgs)  # 假设返回 [B, seq_len]
            gen_seqs_tensor = gen_seqs.clone().detach().to(device)  # 形状：[B, seq_len]
            print("Running discriminator...")
            prob_fake = model.discriminate(gen_seqs_tensor, ast_nodes, ast_structures)
            reward = torch.sigmoid(prob_fake)
            log_probs = F.log_softmax(scores, dim=-1)
            sampled_ids = scores.argmax(dim=-1)
            log_probs_selected = log_probs.gather(2, sampled_ids.unsqueeze(-1)).squeeze(-1)
            reinforce_loss = -torch.mean(log_probs_selected * reward.expand_as(log_probs_selected))

            loss_generator = loss_content + 0.1 * reinforce_loss

            prob_real = model.discriminate(caps_sorted, ast_nodes, ast_structures)
            criterion_adv = nn.BCEWithLogitsLoss()
            loss_disc_fake = criterion_adv(prob_fake.detach(), torch.zeros_like(prob_fake))
            loss_disc_real = criterion_adv(prob_real, torch.ones_like(prob_real))
            loss_discriminator = loss_disc_fake + loss_disc_real

            # 缩放损失以适应梯度累积
            loss_generator = loss_generator / accumulation_steps
            loss_discriminator = loss_discriminator / accumulation_steps

        scaler.scale(loss_generator).backward(retain_graph=True)
        scaler.scale(loss_discriminator).backward()
        
        # 只在累积结束时更新参数
        if (optimizer_step + 1) % accumulation_steps == 0:
            scaler.step(encoderOptimizer)
            scaler.step(decoderOptimizer)
            scaler.step(discOptimizer)
            scaler.update()
        
        optimizer_step += 1

        losses.update(loss_generator.item())
        batch_time.update(time.time() - start)
        start = time.time()

        print(f"Batch {i} completed in {batch_time.val:.3f}s")

        if i % config.print_freq == 0:
            logger = logging.getLogger()
            logger.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                        f'Batch Time: {batch_time.val:.3f}s '
                        f'Loss: {losses.val:.4f} (avg: {losses.avg:.4f}) '
                        f'Enc LR: {get_lr(encoderOptimizer):.6f} '
                        f'Dec LR: {get_lr(decoderOptimizer):.6f} '
                        f'Disc LR: {get_lr(discOptimizer):.6f}')

    return losses.avg

def validate(val_loader, model, criterion, logger):
    model.eval()
    losses = AverageMeter()
    top3_accs = AverageMeter()
    references = []  # 存储真实标签
    hypotheses = []  # 存储预测结果
    
    with torch.no_grad():
        for i, (imgs, caps, caplens, ast_nodes, ast_structures) in enumerate(val_loader):
            # 按 caplens 降序排序批次数据
            sorted_indices = torch.argsort(caplens, descending=True)
            imgs = torch.index_select(imgs, 0, sorted_indices)
            caps = torch.index_select(caps, 0, sorted_indices)
            caplens = torch.index_select(caplens, 0, sorted_indices)
            ast_nodes = torch.index_select(ast_nodes, 0, sorted_indices)
            ast_structures = [ast_structures[i] for i in sorted_indices]

            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            ast_nodes = ast_nodes.to(device)

            # 获取模型预测
            scores, caps_sorted, decode_lengths = model(imgs, caps, caplens)
            targets = caps_sorted[:, 1:]
            
            # 计算 top-3 准确率
            scores_copy = scores.clone()
            # 只计算有效长度内的 top-3 准确率
            top3_acc = 0
            for j, length in enumerate(decode_lengths):
                score_seq = scores_copy[j, :length]  # [length, vocab_size]
                target_seq = targets[j, :length]     # [length]
                _, pred_top3 = score_seq.topk(3, dim=-1)  # [length, 3]
                correct = pred_top3.eq(target_seq.unsqueeze(-1)).any(dim=-1)  # [length]
                top3_acc += correct.float().sum().item()
            top3_acc = top3_acc / float(sum(decode_lengths))
            top3_accs.update(top3_acc, sum(decode_lengths))
            
            # 计算损失
            scores_packed = pack_padded_sequence(scores, decode_lengths, batch_first=True).data
            targets_packed = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
            loss = criterion(scores_packed, targets_packed)
            losses.update(loss.item(), sum(decode_lengths))
            
            # 收集预测结果和真实标签
            _, preds = torch.max(scores, dim=-1)
            for j, length in enumerate(decode_lengths):
                # 获取预测序列（去除填充）
                pred_seq = preds[j][:length].tolist()
                hypotheses.append(pred_seq)
                # 获取真实标签（去除开始标记和填充）
                target_seq = targets[j][:length].tolist()
                references.append([target_seq])  # 每个参考是单元素列表

    # 使用 evaluate 函数评估性能
    score = metrics.evaluate(losses, top3_accs, references, hypotheses, logger)
    return score

def get_lr(optimizer):
    """获取当前学习率"""
    for param_group in optimizer.param_groups:
        return param_group['lr']


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.focal = FocalLoss()
        self.structural_similarity = structural_similarity
        
    def forward(self, pred, target):
        # 交叉熵损失
        ce_loss = self.ce(pred, target)
        # Focal Loss处理类别不平衡
        focal_loss = self.focal(pred, target)
        # 结构相似性损失
        structure_loss = self.structural_similarity(pred, target)
        return ce_loss + 0.5 * focal_loss + 0.3 * structure_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()


def structural_similarity(pred, target):
    """计算结构相似性损失"""
    # 将预测转换为概率分布
    pred_prob = F.softmax(pred, dim=-1)
    # 将目标转换为one-hot编码
    target_one_hot = F.one_hot(target, num_classes=pred.size(-1)).float()
    
    # 计算结构相似性
    mu_x = pred_prob.mean(dim=1)
    mu_y = target_one_hot.mean(dim=1)
    sigma_x = pred_prob.std(dim=1)
    sigma_y = target_one_hot.std(dim=1)
    
    c1 = (0.01 * pred_prob.max())**2
    c2 = (0.03 * pred_prob.max())**2
    
    ssim = (2 * mu_x * mu_y + c1) * (2 * sigma_x * sigma_y + c2) / \
           ((mu_x**2 + mu_y**2 + c1) * (sigma_x**2 + sigma_y**2 + c2))
    
    return 1 - ssim.mean()


def save_checkpoint(state, is_best, checkpoint_dir='checkpoints', final=False):
    """保存检查点
    Args:
        state: 包含模型状态等信息的字典
        is_best: 是否是最佳模型
        checkpoint_dir: 检查点保存目录
        final: 是否为最终模型
    """
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{config.data_name}.pth')
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, f'model_best_{config.data_name}.pth')
        import shutil
        shutil.copyfile(checkpoint_path, best_path)

    if final:
        final_path = os.path.join(checkpoint_dir, f'model_final_{config.data_name}.pth')
        import shutil
        shutil.copyfile(checkpoint_path, final_path)


if __name__ == '__main__':
    main()