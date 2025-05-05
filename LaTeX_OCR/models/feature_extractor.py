import torch
import torch.nn as nn
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe is [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim] if batch_first=False
               Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True (Let's use batch_first=True)
        """
        # x is [batch_size, seq_len, embedding_dim]
        # self.pe is [max_len, 1, d_model] -> need [1, seq_len, d_model]
        pe_for_x = self.pe[:x.size(1)].permute(1, 0, 2)
        x = x + pe_for_x
        return self.dropout(x)


class FeatureExtractorMTL(nn.Module):
    """
    LaTeX Feature Extractor using Multi-Task Learning (MLM + Structure Prediction).
    Uses a shared Transformer Encoder backbone.
    """

    def __init__(self, vocab_size: int, d_model: int = 768, nhead: int = 12, num_encoder_layers: int = 12,
                 dim_feedforward: int = 3072,  # four times d_model
                 dropout: float = 0.1, max_seq_len: int = 256, pad_token_id: int = 0, max_bracket_depth: int = 10):
        """
        Initializes the multi-task feature extractor.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int): Dimension of the model embeddings and Transformer layers.
            nhead (int): Number of attention heads in the Transformer encoder.
            num_encoder_layers (int): Number of layers in the Transformer encoder.
            dim_feedforward (int): Dimension of the feedforward network model in nn.TransformerEncoderLayer.
            dropout (float): Dropout probability.
            max_seq_len (int): Maximum sequence length for positional encoding.
            pad_token_id (int): ID of the PAD token for calculating non-padded loss.
            max_bracket_depth (int): Maximum bracket nesting depth to predict (used for clamping).
        """
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.max_bracket_depth = max_bracket_depth

        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)

        # Shared Transformer Encoder Backbone
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        # Task Head
        # a) MLM Head   often includes a dense layers -> activations ->layer norm before the final prediction layer
        self.mlm_transform_dense = nn.Linear(d_model, d_model)
        self.mlm_activation = nn.GELU()
        self.mlm_layer_norm = nn.LayerNorm(d_model)
        # the actual prediction layer (decoder bias is often tied/set to zero)
        self.mlm_decoder = nn.Linear(d_model, vocab_size, bias=False)
        # optional: tie weights between token_embedding and mlm_decoder
        # self.mlm_decoder.weight = self.token_embedding.weight

        # b) Structure prediction Head
        self.structure_head = nn.Linear(d_model, 1)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor,  # [B, seq_len]
                attention_mask: torch.Tensor,  # [B, seq_len], 1 for real tokens, 0 for padding
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
                Forward pass for multi-task training.

                Args:
                    input_ids (torch.Tensor): Input token IDs, potentially with [MASK] tokens. Shape: [B, seq_len].
                    attention_mask (torch.Tensor): Mask indicating padding tokens. Shape: [B, seq_len].

                Returns:
                    tuple[torch.Tensor, torch.Tensor]:
                        - mlm_logits (torch.Tensor): Logits for the MLM task. Shape: [B, seq_len, vocab_size].
                        - structure_logits (torch.Tensor): Logits/predictions for the structure task (depth). Shape: [B, seq_len, 1].
                """
        # Embedding
        embedding_output = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        embedding_output = self.pos_encoder(embedding_output)

        # Shared Transformer Encoder
        # The TransformerEncoder expects a mask where True indicates positions to *ignore*
        # Our attention_mask is 1 for real tokens, 0 for padding. We need the inverse.
        # Shape expected by src_key_padding_mask is [B, seq_len]
        padding_mask = (attention_mask == 0)  # True for padding positions

        encoder_output = self.transformer_encoder(src=embedding_output, src_key_padding_mask=padding_mask)

        # Task Heads
        # a) MLM Head
        mlm_transformed = self.mlm_transform_dense(encoder_output)
        mlm_activated = self.mlm_activation(mlm_transformed)
        mlm_normed = self.mlm_layer_norm(mlm_activated)
        mlm_logits = self.mlm_decoder(mlm_normed)
        # mlm_logits shape: [B, seq_len, vocab_size]

        # b) Structure Head
        structure_logits = self.structure_head(encoder_output)
        # structure_logits shape: [B, seq_len, 1] (raw depth prediction)

        return mlm_logits, structure_logits

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
               pooling_strategy: str = 'mean') -> torch.Tensor:
        """
        Encodes input sequences to get fixed-size feature vectors.
        Used in the RL phase after pre-training. Does NOT use task heads.

        Args:
            input_ids (torch.Tensor): Input token IDs. Shape: [B, seq_len].
            attention_mask (torch.Tensor): Mask indicating padding. Shape: [B, seq_len].
            pooling_strategy (str): How to pool token-level features ('mean', 'cls', 'max').

        Returns:
            torch.Tensor: Pooled feature representation. Shape: [B, d_model].
        """
        embedding_output = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        embedding_output = self.pos_encoder(embedding_output)
        padding_mask = (attention_mask == 0)
        encoder_output = self.transformer_encoder(src=embedding_output, src_key_padding_mask=padding_mask)

        # Pooling
        if pooling_strategy == 'mean':
            # Masked mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_output.size()).float()
            sum_embeddings = torch.sum(encoder_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)    # Prevent division by zero
            pooled_output = sum_embeddings / sum_mask
        elif pooling_strategy == 'cls':      # Assumes the first token ([CLS]) contains the pooled representation
            if input_ids.shape[1] == 0:
                return torch.zeros(input_ids.shape[0], self.d_model, device=input_ids.device)
            pooled_output = encoder_output[:, 0, :]
        elif pooling_strategy == 'max':
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_output.size()).float()
            encoder_output[input_mask_expanded == 0] = -1e9     # Set padding tokens to a very small value before max pooling
            pooled_output = torch.max(encoder_output, 1)[0]
        else:
            raise  ValueError(f"Unsupported Pooling Strategy: {pooling_strategy}")

        return pooled_output


