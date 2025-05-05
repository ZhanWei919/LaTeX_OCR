import torch
import yaml
import json
import os
from PIL import Image
from torchvision import transforms
import sys
import gradio as gr
import time

# --- 项目路径设置 ---
current_script_path = os.path.abspath(__file__)
project_root = os.path.dirname(current_script_path)
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"添加到 sys.path: {project_root}")

# --- 导入自定义模块 ---
try:
    from models.ocr_model import OCRModel
    from models.utils import load_checkpoint, sequence_to_text
except ImportError as e:
    print(f"错误: 无法导入自定义模块。 Error: {e}")
    sys.exit(1)

# --- 全局加载资源 ---
print("开始加载全局资源...")

# --- 配置路径 ---
CONFIG_PATH = os.path.join(project_root, "configs", "train_rl_config.yaml")
VOCAB_PATH = os.path.join(project_root, "data", "vocab.json")
CHECKPOINT_PATH = os.path.join(project_root, "checkpoints", "model_best.pth.tar")  # 修正路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_GENERATION_METHOD = 'beam'
DEFAULT_BEAM_WIDTH = 5

# 检查文件是否存在
if not all(os.path.exists(p) for p in [CONFIG_PATH, VOCAB_PATH, CHECKPOINT_PATH]):
    print("错误: 必要的配置文件、词汇表或检查点文件未找到！请检查路径。")
    LOAD_ERROR = True
else:
    LOAD_ERROR = False

# --- 加载配置 ---
try:
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    data_config = config["data"]
    model_config = config["model"]
    eval_config = config["evaluation"]
except Exception as e:
    print(f"错误: 加载配置失败: {e}")
    LOAD_ERROR = True

# --- 加载词汇表 ---
try:
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    rev_vocab = {v: k for k, v in vocab.items()}
    vocab_size = len(vocab)
    pad_token_id = vocab.get('<PAD>', 0)
    sos_token_id = vocab.get('<SOS>', 1)
    eos_token_id = vocab.get('<EOS>', 2)
except Exception as e:
    print(f"错误: 加载词汇表失败: {e}")
    LOAD_ERROR = True

# --- 定义图像预处理 ---
try:
    img_transform = transforms.Compose([
        transforms.Resize((data_config["image_height"], data_config["image_width"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config["image_mean"], std=data_config["image_std"]),
    ])
except Exception as e:
    print(f"错误: 定义图像转换失败: {e}")
    LOAD_ERROR = True

# --- 初始化并加载模型 ---
model = None
if not LOAD_ERROR:
    try:
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
            vit_pretrained=False
        ).to(DEVICE)
        load_checkpoint(CHECKPOINT_PATH, model, optimizer=None, map_location=DEVICE)
        model.eval()
        print(f"模型加载成功并移至 {DEVICE}。")
    except Exception as e:
        print(f"错误: 初始化或加载模型失败: {e}")
        LOAD_ERROR = True
        model = None

print("全局资源加载完成。")

# --- 定义 Gradio 的预测函数 ---
def predict_image_latex(input_image: Image.Image, generation_method: str, beam_width: int) -> str:
    if LOAD_ERROR or model is None:
        return "错误：模型未能成功加载，请检查后台日志。"

    start_time = time.time()
    try:
        image = input_image.convert('RGB')
        image_tensor = img_transform(image)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            generated_ids = model.generate(
                image_tensor,
                max_len=data_config["max_seq_len"],
                method=generation_method,
                beam_width=beam_width if generation_method == 'beam' else eval_config.get("beam_width", 5),
                length_penalty=eval_config.get("length_penalty", 0.7)
            )
            generated_ids_list = generated_ids.squeeze(0).cpu().tolist()

        predicted_tokens = sequence_to_text(
            generated_ids_list, rev_vocab, vocab,
            eos_token='<EOS>', pad_token='<PAD>', sos_token='<SOS>'
        )
        predicted_latex = " ".join(predicted_tokens)

        end_time = time.time()
        print(f"单次预测完成，耗时: {end_time - start_time:.2f} 秒。")
        return predicted_latex

    except Exception as e:
        print(f"错误: 预测过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return f"预测出错: {e}"

# --- 创建 Gradio 界面 ---
print("正在创建 Gradio 界面...")
with gr.Blocks() as demo:
    gr.Markdown("# 图像到 LaTeX 转换器 (OCR)")
    gr.Markdown("上传一个数学公式的图片，模型将尝试预测其对应的 LaTeX 代码。")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="上传公式图片")
            method = gr.Dropdown(choices=['beam', 'greedy'], value=DEFAULT_GENERATION_METHOD, label="生成方法")
            beam_width = gr.Slider(minimum=1, maximum=20, step=1, value=DEFAULT_BEAM_WIDTH, label="Beam Width")


        with gr.Column():
            output = gr.Textbox(label="预测的 LaTeX 代码", lines=5)


    submit_btn = gr.Button("生成")
    submit_btn.click(
        fn=predict_image_latex,
        inputs=[image_input, method, beam_width],
        outputs=output


    )

demo.launch()

print("正在启动 Gradio 应用...")
