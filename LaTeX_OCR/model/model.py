import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from torchvision.models import resnet18, ResNet18_Weights

class EncoderViT(nn.Module):
    def __init__(self, input_channels=1, patch_size=None, d_model=512, nhead=8, num_layers=6):
        super(EncoderViT, self).__init__()
        # 初始化 ResNet18，不使用 torch.compile
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 根据输入通道数调整 conv1
        if input_channels != 3:
            # 用新的单通道 conv1 替换默认的 3 通道 conv1
            self.resnet.conv1 = nn.Conv2d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )
        self.resnet.fc = nn.Identity()  # 移除全连接层
        self.d_model = d_model
        self.patch_size = patch_size if patch_size else 2  # 默认 patch_size 为 2
        self.patch_embedding = nn.Conv2d(512, d_model, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_encoding = PositionalEncoding1D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=2048, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x):
        if x.max() > 1:
            x = (x / 127.5) - 1.0  # 归一化到 [-1, 1]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)  # [B, 512, H/16, W/16]
        # 动态调整 patch_size
        _, _, H, W = x.size()
        if self.patch_size > min(H, W):
            self.patch_size = min(H, W)
            self.patch_embedding = nn.Conv2d(512, self.d_model, kernel_size=self.patch_size, stride=self.patch_size).to(x.device)
        x = self.patch_embedding(x)  # [B, d_model, H/patch_size, W/patch_size]
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)  # [B, seq_len, d_model]
        x = self.pos_encoding(x)
        x = self.transformer(x)  # [B, seq_len, d_model]
        return x

# 1D Sinusoidal Positional Encoding
class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding1D, self).__init__()
        self.d_model = d_model
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B, seq_len, d_model = x.size()
        return x + self.pe[:, :seq_len, :]

# 解码器：注意力驱动的序列生成模块，支持教师强制训练与推理模式
class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, max_seq_length=160, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding1D(d_model)  # Assume this is defined elsewhere
        # Define the decoder layer with batch_first=True
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True  # Key change
        )
        # Initialize the TransformerDecoder with batch_first=True
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_seq_length = max_seq_length
        self.dropout = nn.Dropout(dropout)

    def forward(self, encoder_out, captions, caption_lengths):
        B, seq_len, _ = encoder_out.size()
        captions = captions[:, :-1]  # Remove the last token
        cap_len = (caption_lengths - 1).tolist()
        embed = self.embedding(captions)
        embed = self.pos_encoding(embed)
        # Generate the subsequent mask
        mask = nn.Transformer.generate_square_subsequent_mask(captions.size(1)).to(captions.device)
        # Pass through the transformer with the mask
        output = self.transformer(embed, encoder_out, tgt_mask=mask)
        scores = self.fc(self.dropout(output))
        return scores, captions, cap_len

    def inference(self, encoder_out, beam_size=5):
        B = encoder_out.size(0)  # 获取批次大小，支持批量推理
        device = encoder_out.device
        start_token = torch.zeros(B, 1, dtype=torch.long, device=device)  # <start> token，形状 [B, 1]
        embed = self.embedding(start_token)
        embed = self.pos_encoding(embed)
        generated = [start_token.squeeze(1).tolist()]  # 初始 token 列表
        for _ in range(self.max_seq_length):
            mask = nn.Transformer.generate_square_subsequent_mask(embed.size(1)).to(device)
            output = self.transformer(embed, encoder_out, tgt_mask=mask)
            logits = self.fc(output[:, -1, :])  # 取最后一个时间步的输出
            next_token = logits.argmax(-1)  # [B]
            generated.append(next_token.tolist())
            if all(token == 1 for token in next_token):  # 如果所有样本都生成 <end>，停止
                break
            embed_next = self.embedding(next_token.unsqueeze(1))  # [B, 1, d_model]
            embed = torch.cat([embed, embed_next], dim=1)  # 拼接新 token
            embed = self.pos_encoding(embed)
        # 将 generated 转换为 [B, seq_len] 的张量
        generated = torch.tensor(generated, device=device).T  # [B, seq_len]
        return generated, None



# 1. Tree-LSTM：对树结构中单个节点进行编码
class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(TreeLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        # 计算输入门、输出门、更新门（合并计算）
        self.iou = nn.Linear(input_dim + hidden_dim, 3 * hidden_dim)
        # 计算遗忘门（单独计算）
        self.f = nn.Linear(input_dim + hidden_dim, hidden_dim)

    def forward(self, node_embed, children_h, children_c):
        """
        Args:
            node_embed: 当前节点的嵌入 [B, input_dim]
            children_h: 子节点的隐藏状态 [B, num_children, hidden_dim]
            children_c: 子节点的细胞状态 [B, num_children, hidden_dim]
        Returns:
            h: 当前节点的隐藏状态 [B, hidden_dim]
            c: 当前节点的细胞状态 [B, hidden_dim]
        """
        batch_size = node_embed.size(0)
        if children_h.size(1) > 0:
            children_h_sum = children_h.sum(dim=1)  # [B, hidden_dim]
        else:
            children_h_sum = torch.zeros(batch_size, self.hidden_dim, device=node_embed.device)
        combined = torch.cat([node_embed, children_h_sum], dim=1)  # [B, input_dim + hidden_dim]
        iou = self.iou(combined)  # [B, 3 * hidden_dim]
        i, o, u = torch.split(iou, self.hidden_dim, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)
        if children_h.size(1) > 0:
            # 遗忘门因子
            f = torch.sigmoid(self.f(combined))  # [B, hidden_dim]
            children_c_sum = (f * children_c).sum(dim=1)  # [B, hidden_dim]
        else:
            children_c_sum = torch.zeros(batch_size, self.hidden_dim, device=node_embed.device)
        c = i * u + children_c_sum  # [B, hidden_dim]
        h = o * torch.tanh(c)       # [B, hidden_dim]
        return h, c

# 2. AST 编码器：利用 Tree-LSTM 对整个 AST 进行编码
class ASTEncoder(nn.Module):
    def __init__(self, ast_vocab_size, ast_embed_dim, ast_hidden_dim):
        super(ASTEncoder, self).__init__()
        self.embedding = nn.Embedding(ast_vocab_size, ast_embed_dim)
        self.tree_lstm = TreeLSTM(ast_embed_dim, ast_hidden_dim)

    def forward(self, ast_nodes, ast_structure):
        """
        Args:
            ast_nodes: [B, max_num_nodes] 张量，每个样本的AST节点（已经做过padding）。
            ast_structure: 长度为 B 的列表，每个元素是该样本的树结构，
                           每个样本是一个长度为 n 的列表，其中每个元素是该节点的子节点索引列表。
        Returns:
            root_h: [B, ast_hidden_dim] 每个样本根节点的表示。
        """
        embeddings = self.embedding(ast_nodes)  # [B, max_num_nodes, ast_embed_dim]
        B, max_n, _ = embeddings.size()
        root_reps = []  # 存放每个样本的根节点表示
        for b in range(B):
            # 对于每个样本，只处理有效节点，原始节点数
            n = len(ast_structure[b])
            emb_b = embeddings[b, :n, :]  # 取有效节点的嵌入
            h = torch.zeros((n, self.tree_lstm.hidden_dim), device=embeddings.device)
            c = torch.zeros((n, self.tree_lstm.hidden_dim), device=embeddings.device)
            # 按节点顺序从叶到根递归计算（假设节点顺序已满足从根到叶的顺序）
            for node_idx in reversed(range(n)):
                children_idx = ast_structure[b][node_idx]
                if children_idx:
                    children_idx_tensor = torch.tensor(children_idx, dtype=torch.long, device=embeddings.device)
                    children_h = h.index_select(0, children_idx_tensor).unsqueeze(0)  # [1, num_children, hidden_dim]
                    children_c = c.index_select(0, children_idx_tensor).unsqueeze(0)
                else:
                    children_h = torch.zeros((1, 1, self.tree_lstm.hidden_dim), device=embeddings.device)
                    children_c = torch.zeros((1, 1, self.tree_lstm.hidden_dim), device=embeddings.device)
                node_embed = emb_b[node_idx].unsqueeze(0)  # [1, ast_embed_dim]
                h_node, c_node = self.tree_lstm(node_embed, children_h, children_c)
                h[node_idx] = h_node.squeeze(0)
                c[node_idx] = c_node.squeeze(0)
            # 假设根节点在 index 0
            root_reps.append(h[0])
        return torch.stack(root_reps, dim=0)

# 3. 跨模态特征融合：使用交叉注意力来融合 token 与 AST 特征
class CrossAttentionFusion(nn.Module):
    def __init__(self, token_dim, ast_dim, fused_dim):
        super(CrossAttentionFusion, self).__init__()
        self.query_proj = nn.Linear(token_dim, fused_dim)
        self.key_proj = nn.Linear(ast_dim, fused_dim)
        self.value_proj = nn.Linear(ast_dim, fused_dim)
        self.scale = torch.sqrt(torch.tensor(fused_dim, dtype=torch.float32))

    def forward(self, token_context, ast_context):
        """
        Args:
            token_context: [B, token_dim] Token 序列全局表示
            ast_context: [B, ast_dim] AST 表示（根节点特征）
        Returns:
            fused_out: [B, token_dim + fused_dim] 融合后的特征
        """
        Q = self.query_proj(token_context)      # [B, fused_dim]
        K = self.key_proj(ast_context)            # [B, fused_dim]
        V = self.value_proj(ast_context)          # [B, fused_dim]
        scores = (Q * K) / self.scale             # [B, fused_dim]
        weights = torch.sigmoid(scores)           # [B, fused_dim]（相当于注意力权重）
        fused = weights * V                       # [B, fused_dim]
        # 将原始 token_context 与经过注意力调制的 AST 特征拼接
        fused_out = torch.cat([token_context, fused], dim=1)
        return fused_out



class Discriminator(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim,
                 ast_vocab_size, ast_embed_dim, ast_hidden_dim,
                 fusion_dim, dropout=0.5):
        super(Discriminator, self).__init__()
        # Token 序列编码：嵌入层 + 双向 LSTM + 注意力池化
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.bilstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)
        
        # AST 编码器
        self.ast_encoder = ASTEncoder(ast_vocab_size, ast_embed_dim, ast_hidden_dim)
        # 保存 AST 隐藏层维度
        self.ast_hidden_dim = ast_hidden_dim
        
        # 特征融合模块：交叉注意力融合 token 与 AST 特征
        # token_context 的维度： hidden_dim * 2
        # ast_context 的维度： ast_hidden_dim
        self.fusion = CrossAttentionFusion(token_dim=hidden_dim * 2,
                                           ast_dim=ast_hidden_dim,
                                           fused_dim=fusion_dim)
        
        # 分类头，根据融合后的特征输出 logits
        # 融合后特征维度： hidden_dim*2 + fusion_dim
        self.fc_out = nn.Linear(hidden_dim * 2 + fusion_dim, 1)

    def forward(self, seqs, ast_nodes=None, ast_structure=None):
        """
        Args:
            seqs: [B, seq_len] Token 序列
            ast_nodes: [B, num_nodes] AST 节点序列（整数表示），可选；若为 None，则使用零张量
            ast_structure: 树结构信息；格式：长度为 num_nodes 的列表，每个元素为该节点的子节点索引列表，可选
        Returns:
            logits: [B, 1] 判别器输出 logits
        """
        # Token 序列编码
        emb = self.embedding(seqs)              # [B, seq_len, embed_dim]
        out, _ = self.bilstm(emb)                # [B, seq_len, hidden_dim*2]
        attn_scores = self.attention_fc(out)     # [B, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, seq_len, 1]
        token_context = torch.sum(attn_weights * out, dim=1)  # [B, hidden_dim*2]
        
        # 如果未提供 AST 信息，则使用全 0 张量作为 AST 上下文
        if ast_nodes is None or ast_structure is None:
            batch_size = seqs.size(0)
            ast_context = torch.zeros(batch_size, self.ast_hidden_dim, device=seqs.device)
        else:
            ast_context = self.ast_encoder(ast_nodes, ast_structure)  # [B, ast_hidden_dim]
        
        # 融合 token 与 AST 特征
        fused_context = self.fusion(token_context, ast_context)  # [B, hidden_dim*2 + fusion_dim]
        
        # 最后输出判别 logits
        logits = self.fc_out(fused_context)  # [B, 1]
        return logits



class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, encoder_dim, disc_hidden_dim, fusion_dim,
                 ast_vocab_size, ast_embed_dim, ast_hidden_dim, max_seq_length=160, dropout=0.5):
        super(Model, self).__init__()
        self.encoder = EncoderViT(input_channels=1, d_model=encoder_dim)
        self.decoder = TransformerDecoder(vocab_size, encoder_dim, 8, 6, max_seq_length, dropout)
        self.discriminator = Discriminator(vocab_size, embed_dim, disc_hidden_dim, ast_vocab_size, ast_embed_dim, ast_hidden_dim, fusion_dim, dropout)

    def forward(self, images, captions, caption_lengths):
        encoder_out = self.encoder(images)
        scores, caps_sorted, decode_lengths = self.decoder(encoder_out, captions, caption_lengths)
        return scores, caps_sorted, decode_lengths

    def generate(self, images, beam_size=5):
        encoder_out = self.encoder(images)
        return self.decoder.inference(encoder_out, beam_size)

    def discriminate(self, seqs, ast_nodes, ast_structure):
        return self.discriminator(seqs, ast_nodes, ast_structure)