import time

import torch
import yaml
import json
import argparse
import os
from PIL import Image
from torchvision import transforms
import sys

# --- 项目路径设置 (确保能找到 models 目录) ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_script_path))
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"添加到 sys.path: {project_root}")

# --- 导入自定义模块 ---
try:
    from models.ocr_model import OCRModel
    from models.utils import load_checkpoint, sequence_to_text
except ImportError as e:
    print(f"错误: 无法导入自定义模块。请确保脚本相对于项目根目录的位置正确，或者已将项目根目录添加到 PYTHONPATH。 Error: {e}")
    sys.exit(1)

def predict_latex(args):
    """
    加载模型并对单个图像进行 LaTeX 预测。
    """
    # --- 1. 加载配置 ---
    try:
        with open(args.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("配置加载成功。")
    except Exception as e:
        print(f"错误: 无法加载配置文件 {args.config_path}: {e}")
        return None

    # --- 2. 加载词汇表 ---
    try:
        with open(args.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)
        rev_vocab = {v: k for k, v in vocab.items()}
        vocab_size = len(vocab)
        # 从 config 或 vocab 获取特殊 token ID (模型初始化需要)
        data_config = config["data"] # 获取数据配置部分
        model_config = config["model"] # 获取模型配置部分
        pad_token_id = vocab.get('<PAD>', 0)
        sos_token_id = vocab.get('<SOS>', 1)
        eos_token_id = vocab.get('<EOS>', 2)
        print(f"词汇表加载: {vocab_size} tokens.")
    except Exception as e:
        print(f"错误: 无法加载词汇表 {args.vocab_path}: {e}")
        return None

    # --- 3. 设置设备 ---
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"使用设备: {device}")

    # --- 4. 初始化并加载模型 ---
    try:
        # 使用配置文件中的参数初始化模型
        model = OCRModel(
            vocab_size=vocab_size,
            d_model=model_config["d_model"],
            decoder_nhead=model_config["decoder_nhead"],
            decoder_layers=model_config["decoder_layers"],
            decoder_dim_feedforward=model_config["decoder_dim_feedforward"],
            decoder_dropout=model_config["decoder_dropout"],
            max_seq_len=data_config["max_seq_len"],
            pad_token_id=pad_token_id,
            sos_token_id=sos_token_id,
            eos_token_id=eos_token_id,
            vit_model_name=model_config["vit_model_name"],
            vit_pretrained=False # 推理时通常不需要重新加载预训练权重，我们会加载自己的 checkpoint
        ).to(device)
        print("OCR 模型结构初始化完成。")

        # --- !! 关键：加载检查点，但不需要优化器状态 !! ---
        checkpoint_info = load_checkpoint(args.checkpoint_path, model, optimizer=None, map_location=device)
        if not checkpoint_info and not hasattr(model, 'state_dict'): # 简单检查模型是否加载成功
             print("错误：模型状态字典未能从检查点加载。")
             return None
        # -------------------------------------------------

        model.eval() # 设置为评估模式！
        print(f"模型权重从 {args.checkpoint_path} 加载成功。")

    except Exception as e:
        print(f"错误: 初始化或加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None

    # --- 5. 定义图像预处理 ---
    # 使用配置文件中的图像参数
    img_transform = transforms.Compose([
        transforms.Resize((data_config["image_height"], data_config["image_width"])),
        # 灰度图像处理: 如果你的模型是用灰度图训练的，需要转换
        # transforms.Grayscale(num_output_channels=3), # 如果 ViT 需要 3 通道输入
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["image_mean"], std=data_config["image_std"]),
    ])
    print("图像预处理流程定义完成。")

    # --- 6. 加载和预处理输入图像 ---
    try:
        image = Image.open(args.image_path).convert('RGB') # 确保是 RGB
        image_tensor = img_transform(image)
        # 添加 batch 维度 (模型通常需要 B x C x H x W)
        image_tensor = image_tensor.unsqueeze(0).to(device)
        print(f"图像 {args.image_path} 加载并预处理完成。")
    except FileNotFoundError:
        print(f"错误: 图像文件未找到于 {args.image_path}")
        return None
    except Exception as e:
        print(f"错误: 加载或预处理图像失败: {e}")
        return None

    # --- 7. 执行推理 ---
    print(f"开始使用 {args.method} 方法生成 LaTeX...")
    eval_config = config["evaluation"] # 获取评估配置以获取 beam width 等参数
    start_time = time.time()
    with torch.no_grad(): # 推理不需要计算梯度
        # 使用模型的 generate 方法
        generated_ids = model.generate(
            image_tensor,
            max_len=data_config["max_seq_len"],
            method=args.method,
            beam_width=args.beam_width if args.method == 'beam' else eval_config.get("beam_width", 5), # 使用命令行或配置文件的beam width
            length_penalty=eval_config.get("length_penalty", 0.7) # 从配置文件获取长度惩罚
        )
        # generated_ids 是一个包含一个序列的张量, shape: [1, seq_len]
        generated_ids_list = generated_ids.squeeze(0).cpu().tolist() # 移除 batch 维并转为 list
    end_time = time.time()
    print(f"生成完成，耗时: {end_time - start_time:.2f} 秒。")

    # --- 8. 解码输出序列 ---
    try:
        predicted_tokens = sequence_to_text(
            generated_ids_list,
            rev_vocab,
            vocab, # 传递 vocab 用于获取特殊 token
            eos_token='<EOS>',
            pad_token='<PAD>',
            sos_token='<SOS>'
        )
        predicted_latex = " ".join(predicted_tokens) # 将 token 列表连接成字符串
        return predicted_latex
    except Exception as e:
        print(f"错误: 解码输出序列失败: {e}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用训练好的 OCR 模型预测图像的 LaTeX 代码")

    # --- 默认路径设置 (基于脚本相对于项目根目录的位置) ---
    default_config_path = os.path.join(project_root, "configs", "train_rl_config.yaml")
    default_vocab_path = os.path.join(project_root, "data", "vocab.json")
    default_checkpoint_path = os.path.join(project_root, "checkpoints", "model_best.pth.tar")
    # ---------------------------------------------------

    parser.add_argument("--image_path", type=str, required=True,
                        help="输入图像的路径")
    parser.add_argument("--checkpoint_path", type=str, default=default_checkpoint_path,
                        help=f"模型检查点文件的路径 (默认: {default_checkpoint_path})")
    parser.add_argument("--config_path", type=str, default=default_config_path,
                        help=f"训练配置 YAML 文件的路径 (默认: {default_config_path})")
    parser.add_argument("--vocab_path", type=str, default=default_vocab_path,
                        help=f"词汇表 JSON 文件的路径 (默认: {default_vocab_path})")
    parser.add_argument("--device", type=str, default='auto', choices=['auto', 'cuda', 'cpu'],
                        help="运行设备 ('auto', 'cuda', 'cpu')")
    parser.add_argument("--method", type=str, default='beam', choices=['greedy', 'beam'],
                        help="生成方法 ('greedy' 或 'beam')")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam search 的宽度 (仅当 method='beam' 时有效)")

    args = parser.parse_args()

    # --- 检查文件是否存在 ---
    if not os.path.exists(args.image_path):
        print(f"错误: 输入图像文件不存在于 {args.image_path}")
        sys.exit(1)
    if not os.path.exists(args.checkpoint_path):
        print(f"错误: 检查点文件不存在于 {args.checkpoint_path}")
        sys.exit(1)
    if not os.path.exists(args.config_path):
        print(f"错误: 配置文件不存在于 {args.config_path}")
        sys.exit(1)
    if not os.path.exists(args.vocab_path):
        print(f"错误: 词汇表文件不存在于 {args.vocab_path}")
        sys.exit(1)
    # -----------------------

    predicted_latex = predict_latex(args)

    if predicted_latex is not None:
        print("\n" + "="*20 + " 预测结果 " + "="*20)
        print(predicted_latex)
        print("="*50)