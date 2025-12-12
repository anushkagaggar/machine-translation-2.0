import torch
from pathlib import Path
import argparse
import sentencepiece as spm

from src.models.transformer import MTTransformer


# ----------------------------
# Load tokenizer exactly as training
# ----------------------------
def load_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp


# ----------------------------
# Encode input using training rules
# ----------------------------
def encode_src(sp, text, max_len):
    ids = sp.encode(text, out_type=int)
    ids = ids[: max_len - 1] + [2]  # EOS = 2
    return torch.tensor(ids, dtype=torch.long)


# ----------------------------
# Autoregressive greedy decode
# ----------------------------
@torch.no_grad()
def greedy_decode(model, src_tensor, sp, max_len, pad_id=3, bos_id=1, eos_id=2, device="cpu"):
    model.eval()

    src_tensor = src_tensor.unsqueeze(0).to(device)   # (1, S)
    src_pad_mask = (src_tensor == pad_id)

    # encoder output is auto-handled by forward()
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_pad_mask = (generated == pad_id)

        L = generated.size(1)
        tgt_mask = torch.triu(torch.ones((L, L), device=device) == 1, diagonal=1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))

        logits = model(
            src_tensor,
            generated,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask
        )

        # get last step prediction
        next_token = logits[:, -1, :].argmax(dim=-1)

        # append
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        if next_token.item() == eos_id:
            break

    # remove BOS
    out_ids = generated[0].tolist()[1:]
    return sp.decode(out_ids)


# ----------------------------
# Main CLI
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--src", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    ckpt = torch.load(args.model, map_location=device)
    config = ckpt["config"]["model"]

    sp = load_tokenizer("tokenizer/tokenizer.model")

    model = MTTransformer(
        src_vocab_size=config["src_vocab_size"],
        tgt_vocab_size=config["tgt_vocab_size"],
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_encoder_layers=config["encoder_layers"],
        num_decoder_layers=config["decoder_layers"],
        dim_feedforward=config["d_ff"],
        dropout=config["dropout"],
        max_len=config["max_len"],
        pad_id=config["pad_id"],
        tie_embeddings=config["tie_embeddings"]
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print("Model loaded!")

    # encode input
    src_tensor = encode_src(sp, args.src, config["max_len"])

    # decode
    translation = greedy_decode(
        model, src_tensor, sp,
        max_len=config["max_len"],
        pad_id=config["pad_id"]
    )

    print("\n=== Translation ===")
    print("Input: ", args.src)
    print("Output:", translation)


if __name__ == "__main__":
    main()
