# src/models/transformer.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x

class MTTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int = 8000,
        tgt_vocab_size: int = 8000,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 80,
        pad_id: int = 3,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)

        if tie_embeddings:
            # tie target embedding weights to source embedding weights
            # This ties input embedding of encoder and decoder token embeddings
            # and we will also tie generator weight to src token embedding
            self.tgt_tok_emb.weight = self.src_tok_emb.weight

        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation="relu")
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation="relu")

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # final projection (tie with embedding if requested)
        self.generator = nn.Linear(d_model, tgt_vocab_size, bias=False)
        if tie_embeddings:
            self.generator.weight = self.src_tok_emb.weight

        self.pad_id = pad_id
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src_tokens, src_key_padding_mask):
        # src_tokens: (batch, seq_len)
        x = self.src_tok_emb(src_tokens) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        # nn.Transformer expects (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory  # (seq_len, batch, d_model)

    def decode(self, tgt_tokens, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask):
        y = self.tgt_tok_emb(tgt_tokens) * math.sqrt(self.d_model)
        y = self.pos_enc(y)
        y = y.transpose(0, 1)
        out = self.decoder(
            tgt=y,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return out  # seq_len, batch, d_model

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src_tokens: (batch, src_len)
        tgt_tokens: (batch, tgt_len)
        src_key_padding_mask: (batch, src_len) with True on PAD positions for PyTorch's API
        tgt_key_padding_mask: (batch, tgt_len)
        """
        # convert padding mask to PyTorch expected format: True for PAD
        # memory: seq_len, batch, d_model
        memory = self.encode(src_tokens, src_key_padding_mask)
        dec_out = self.decode(tgt_tokens, memory, tgt_key_padding_mask, src_key_padding_mask, tgt_mask)
        # dec_out: seq_len, batch, d_model
        dec_out = dec_out.transpose(0, 1)  # batch, seq_len, d_model
        logits = self.generator(dec_out)  # batch, seq_len, vocab
        return logits
