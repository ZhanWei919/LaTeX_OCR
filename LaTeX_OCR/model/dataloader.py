import torchvision
import torch
import json
import cv2
import numpy as np
from config import vocab_path, buckets, num_workers, batch_size
from torch.utils.data import Dataset, DataLoader
from model.utils import load_json
from scipy.ndimage import map_coordinates, gaussian_filter

vocab = load_json(vocab_path)

# old_size:图像的原始尺寸，通常是一个元组 (width, height)，表示图像的宽度和高度
# buckets: 一个预定义的尺寸列表，每个元素是一个元组 (width, height)，表示可选的图像尺寸
# ratio: 缩放比例，用于调整图像的尺寸。默认值为 2
def get_new_size(old_size, buckets=buckets, ratio=2):
    # 添加最小尺寸限制
    min_bucket = (32, 32)
    if buckets is None:
        return old_size
    else:
        w, h = old_size[0] / ratio, old_size[1] / ratio
        for (idx, (w_b, h_b)) in enumerate(buckets):
            if w_b >= w and h_b >= h:
                return w_b, h_b, idx
    for (idx, (w_b, h_b)) in enumerate(buckets):
        if w_b >= w and h_b >= h:
            return max(w_b, min_bucket[0]), max(h_b, min_bucket[1]), idx
    return (max(old_size[0], min_bucket[0]), max(old_size[1], min_bucket[1]))


# img_data: 输入的图像数据，通常是一个二维数组（灰度图像）
# 提取图像中的非空白区域，并在其周围添加填充
def data_turn(img_data, pad_size=[8, 8, 8, 8], resize=False):
    """图像预处理和数据增强

    Args:
        img_data: 输入图像
        pad_size: 填充大小
        resize: 是否调整大小
    """
    # 强制下采样至1/3

    h, w = img_data.shape[:2]  # 改为自适应缩放
    scale = max(h / 120, w / 400)  # 保持长宽比
    new_h, new_w = int(h / scale), int(w / scale)

    # 确保不超过最大尺寸
    max_h, max_w = 120, 400
    if new_h > max_h:
        new_w = int(new_w * (max_h / new_h))
        new_h = max_h
    if new_w > max_w:
        new_h = int(new_h * (max_w / new_w))
        new_w = max_w

    img_data = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 随机应用数据增强
    if np.random.random() < 0.3:  # 30%概率应用弹性形变
        img_data = elastic_transform(img_data, alpha=new_w * 0.1, sigma=4)

    if np.random.random() < 0.3:  # 30%概率添加墨迹噪声
        num_strokes = int(new_w * new_h * 0.0005)  # 根据图像大小确定线段数量
        img_data = add_noise_strokes(img_data, num_strokes=num_strokes)

    # 提取非空白区域
    nnz_inds = np.where(img_data != 255)
    if len(nnz_inds[0]) == 0:  # 如果图像全白
        return img_data

    y_min = np.min(nnz_inds[1])
    y_max = np.max(nnz_inds[1])
    x_min = np.min(nnz_inds[0])
    x_max = np.max(nnz_inds[0])
    old_im = img_data[x_min:x_max + 1, y_min:y_max + 1]

    # 添加填充
    top, left, bottom, right = pad_size
    old_size = (old_im.shape[0] + left + right, old_im.shape[1] + top + bottom)
    new_im = np.ones(old_size, dtype=np.uint8) * 255
    new_im[top:top + old_im.shape[0], left:left + old_im.shape[1]] = old_im

    if resize:
        new_size = get_new_size(old_size, buckets)[:2]
        new_im = cv2.resize(new_im, new_size, cv2.INTER_LANCZOS4)

    return new_im


# 将输入的文本转换为模型可以处理的格式
# text 是输入的文本，通常是一个字符串。
# start_type、end_type、pad_type 是可选的输入，分别表示文本的开始符号、结束符号和填充符号，默认分别为 '<start>'、'<end>'、'<pad>'。
def label_transform(text, vocab, start_type='<start>', end_type='<end>', pad_type='<pad>', max_len=160):
    text = text.split()

    # 添加起始和结束标记
    text = [start_type] + text + [end_type]

    # 如果需要填充文本到 max_len
    while len(text) < max_len:
        text += [pad_type]

    # 将每个词语转换为词汇表中的索引
    text = [vocab.get(x, vocab['UNKNOWN']) for x in text]

    return text


# 对输入的图像进行下采样、缩放，并将其转换为 PyTorch 张量格式
def img_transform(img, size, ratio=1):
    new_size = (int(img.shape[1] / ratio), int(img.shape[0] / ratio))
    new_im = cv2.resize(img, new_size, cv2.INTER_LANCZOS4)
    new_im = cv2.resize(new_im, tuple(size))
    if len(new_im.shape) == 3:
        new_im = cv2.cvtColor(new_im, cv2.COLOR_BGR2GRAY)
    new_im = new_im.astype(np.float32)
    mean = np.mean(new_im)
    std = np.std(new_im) or 1
    new_im = (new_im - mean) / std  # 确保正负对称
    new_im = np.expand_dims(new_im, axis=0)
    return torch.from_numpy(new_im)


# 读取图像和标签，并对它们进行预处理
class formuladataset(Dataset):
    # 公式数据集,负责读取图片和标签,同时自动对进行预处理
    # ：param json_path 包含图片文件名和标签的json文件
    # ：param pic_transform,label_transform分别是图片预处理和标签预处理(主要是padding)
    def __init__(self, data_json_path, img_transform=img_transform, label_transform=label_transform, ratio=2,
                 batch_size=batch_size):
        super(formuladataset, self).__init__()
        # 预计算并缓存图像变换结果
        self.cached_images = {}
        self.samples_by_bucket = {}  # 初始化 samples_by_bucket 字典
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.data = load_json(data_json_path)
        self.ratio = ratio
        self.batch_size = batch_size
        self.buckets = buckets

        # 预处理数据
        print('Processing and caching dataset...')
        for img_name, item in self.data.items():
            # 预读取和预处理图像
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.cached_images[img_name] = img
            new_size = get_new_size(item['size'], self.buckets, self.ratio)
            if len(new_size) == 3:
                bucket_idx = new_size[-1]
                if bucket_idx not in self.samples_by_bucket:
                    self.samples_by_bucket[bucket_idx] = []
                self.samples_by_bucket[bucket_idx].append({
                    'img_path': item['img_path'],
                    'caption': item['caption'],
                    'caption_len': item['caption_len'],
                    'size': bucket_idx,
                    'ast_nodes': item.get('ast_nodes', []),
                    'ast_structure': item.get('ast_structure', [])
                })
        
        # 将每个bucket中的样本数量调整为batch_size的整数倍
        self.samples = []
        for bucket_idx, bucket_samples in self.samples_by_bucket.items():
            num_samples = len(bucket_samples)
            # 如果样本数不是batch_size的整数倍，则删除多余的样本
            num_batches = num_samples // batch_size
            if num_batches > 0:
                bucket_samples = bucket_samples[:num_batches * batch_size]
                self.samples.extend(bucket_samples)
        
        # 按bucket大小排序，确保同一batch中的样本使用相同的bucket
        self.samples.sort(key=lambda x: x['size'])
        print(f'Processed {len(self.samples)} samples')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 读取图片
        img = cv2.imread(item['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 处理图片
        if self.img_transform is not None:
            img = self.img_transform(img, size=self.buckets[item['size']], ratio=self.ratio)

        # 处理标签
        caption = item['caption']
        if self.label_transform is not None:
            caption = self.label_transform(caption, vocab)
            caption = torch.LongTensor(caption)

        ast_nodes_raw = item.get('ast_nodes', [])
        if len(ast_nodes_raw) == 0:
            ast_nodes_tensor = torch.LongTensor([0])
            ast_structure = [[ ]]  # 表示只有一个节点且没有子节点
        else:
            ast_nodes_tensor = torch.LongTensor(item['ast_nodes'])
            ast_structure = item['ast_structure']

        return {
            "image": img,
            "caption_indices": caption,
            "caption_len": torch.tensor(item['caption_len']),
            "ast_nodes": ast_nodes_tensor,
            "ast_structure": ast_structure
        }

def elastic_transform(image, alpha=40, sigma=4, random_state=None):
    """弹性形变

    Args:
        image: 输入图像
        alpha: 形变强度
        sigma: 高斯核大小
        random_state: 随机种子
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    distored_image = map_coordinates(image, indices, order=1, mode='reflect')
    return distored_image.reshape(shape)


def add_noise_strokes(image, num_strokes=10, max_length=20, thickness=2):
    """添加模拟墨迹噪声

    Args:
        image: 输入图像
        num_strokes: 添加的线段数量
        max_length: 最大线段长度
        thickness: 线段粗细
    """
    result = image.copy()
    h, w = image.shape[:2]

    for _ in range(num_strokes):
        # 随机起点
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        # 随机长度和角度
        length = np.random.randint(5, max_length)
        angle = np.random.uniform(0, 2 * np.pi)

        # 计算终点
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))

        # 确保终点在图像内
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        # 绘制线段
        cv2.line(result, (x1, y1), (x2, y2), 0, thickness)

    return result


def collate_fn(batch):
    """
    假设每个 batch 元素是一个元组，依次包含：
    (image, caption_indices, caption_len, ast_nodes, ast_structure)
    """
    images = []
    captions = []
    caplens = []
    ast_nodes_list = []
    ast_structures = []
    for item in batch:
        images.append(item["image"])
        captions.append(item["caption_indices"])
        caplens.append(item["caption_len"])
        ast_nodes_list.append(item["ast_nodes"])
        ast_structures.append(item["ast_structure"])
    
    images_tensor = torch.stack(images, dim=0)
    padded_caps = torch.nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=0)
    lengths_tensor = torch.tensor(caplens, dtype=torch.long)
    padded_ast_nodes = torch.nn.utils.rnn.pad_sequence(ast_nodes_list, batch_first=True, padding_value=0)
    
    return images_tensor, padded_caps, lengths_tensor, padded_ast_nodes, ast_structures
