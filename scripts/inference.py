# scripts/inference.py
import torch
import argparse
from src.models.transformer import MTTransformer
from src.data.dataset import load_sentencepiece_tokenizer


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    model_cfg = cfg["model"]

    # Build model
    model = MTTransformer(
        src_vocab_size=model_cfg["src_vocab_size"],
        tgt_vocab_size=model_cfg["tgt_vocab_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_encoder_layers=model_cfg["encoder_layers"],
        num_decoder_layers=model_cfg["decoder_layers"],
        dim_feedforward=model_cfg["d_ff"],
        dropout=model_cfg["dropout"],
        max_len=model_cfg["max_len"],
        pad_id=model_cfg["pad_id"],
        tie_embeddings=model_cfg["tie_embeddings"]
    ).to(device)

    # Load weights
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def encode_text(sp, text, pad_id, max_len):
    ids = sp.encode(text, out_type=int)
    ids = ids[:max_len - 1] + [2]  # <eos> at end
    return torch.tensor(ids)


def greedy_decode(model, sp, src_tensor, cfg, device):
    pad_id = cfg["model"]["pad_id"]
    max_len = cfg["model"]["max_len"]

    src = src_tensor.unsqueeze(0).to(device)
    src_pad_mask = (src == pad_id)

    # start sequence with <bos>
    tgt = torch.tensor([[1]], device=device)  # <bos>

    for _ in range(max_len):
        tgt_pad_mask = (tgt == pad_id)

        tgt_seq_len = tgt.size(1)
        causal = torch.triu(
            torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1,
            diagonal=1,
        )
        causal = causal.float().masked_fill(causal, float('-inf'))

        logits = model(
            src,
            tgt,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=causal
        )

        next_token = logits[0, -1].argmax().item()
        tgt = torch.cat([tgt, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == 2:  # <eos>
            break

    ids = tgt[0].tolist()
    return sp.decode(ids[1:])  # remove <bos>


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--src", required=True, help="Input Japanese or English sentence")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load tokenizer
    sp = load_sentencepiece_tokenizer("tokenizer/tokenizer.model")

    # Load checkpoint + model
    model, cfg = load_checkpoint(args.model, device)
    print("Model loaded!")

    # Encode input
    pad_id = cfg["model"]["pad_id"]
    max_len = cfg["model"]["max_len"]
    src_tensor = encode_text(sp, args.src, pad_id, max_len)

    # Translate
    translation = greedy_decode(model, sp, src_tensor, cfg, device)
    print("\n=== Translation ===")
    print("Input: ", args.src)
    print("Output:", translation)


if __name__ == "__main__":
    main()
