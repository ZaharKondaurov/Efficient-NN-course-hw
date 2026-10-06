"""Transformer language model: standard / ZerO / ReZero encoder layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .inits import init_transformer_layers


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class ReZeroEncoderLayer(nn.Module):
    """
    ReZero Transformer layer (Bachlechner et al., 2020; majumderb/rezero RZTX).

    No LayerNorm; one shared residual gate α (resweight) for attention and FFN:
        x <- x + α · Dropout(Attn(x))
        x <- x + α · Dropout(FFN(x))
    with α = 0 at initialization.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        del is_causal  # causal masking comes from src_mask (as in majumderb/rezero)
        src2 = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )[0]
        src = src + self.dropout1(src2) * self.resweight
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2) * self.resweight
        return src


class TransformerLM(nn.Module):
    """
    Word-level LM as in the PyTorch ``word_language_model`` example.
    Input/output layout: (seq_len, batch).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 200,
        nhead: int = 2,
        d_hid: int = 200,
        num_layers: int = 2,
        dropout: float = 0.2,
        init: str = "zero",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        if init == "rezero":
            layer = ReZeroEncoderLayer(d_model, nhead, d_hid, dropout)
            # No final LayerNorm: ReZero replaces LayerNorm with α-gated residuals.
            self.encoder = nn.TransformerEncoder(
                layer, num_layers, norm=None, enable_nested_tensor=False
            )
        else:
            layer = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
            self.encoder = nn.TransformerEncoder(
                layer, num_layers, enable_nested_tensor=False
            )

        self.decoder = nn.Linear(d_model, vocab_size)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.uniform_(self.decoder.weight, -0.1, 0.1)
        nn.init.zeros_(self.decoder.bias)
        init_transformer_layers(self.encoder.layers, init)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        mask = nn.Transformer.generate_square_subsequent_mask(src.size(0), device=src.device)
        x = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        x = self.encoder(x, mask=mask, is_causal=True)
        return self.decoder(x)
