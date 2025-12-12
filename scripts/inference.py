# scripts/inference.py
import torch
import argparse
import sentencepiece as spm
from pathlib import Path
from src.models.transformer import MTTransformer
import math

# =====================
# Configurable debug flag
# =====================
DEBUG_DEVICES = False  # set True to print device info per step

# =====================
# Helpers
# =====================
def load_tokenizer(path: str):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp

def encode_src(sp, text: str, max_len: int, eos_id: int = 2):
    ids = sp.encode(text, out_type=int)
    ids = ids[: max_len - 1] + [eos_id]
    return torch.tensor(ids, dtype=torch.long)  # CPU tensor by default

# =====================
# Greedy decode — ALL tensors explicitly moved to device
# =====================
@torch.no_grad()
def greedy_decode(model, src_tensor_cpu, sp, cfg_model, device="cpu"):
    """
    src_tensor_cpu: 1D CPU tensor of token ids (no batch dim)
    sp: SentencePieceProcessor
    cfg_model: model config dict (contains pad_id, bos/eos, max_len)
    """
    model.eval()
    pad_id = cfg_model.get("pad_id", 3)
    bos_id = cfg_model.get("bos_id", 1) if "bos_id" in cfg_model else 1
    eos_id = cfg_model.get("eos_id", 2) if "eos_id" in cfg_model else 2
    max_len = cfg_model.get("max_len", 80)

    # --- Move source to device and add batch dim
    src = src_tensor_cpu.to(device)          # ensure on device
    src = src.unsqueeze(0)                   # (1, S) on device

    # --- Build src padding mask ON DEVICE
    src_pad_mask = (src == pad_id).to(device)  # same dtype & device as model

    # Debug: show devices
    if DEBUG_DEVICES:
        print("DEBUG: src device", src.device)
        print("DEBUG: model device", next(model.parameters()).device)
        print("DEBUG: src_pad_mask device", src_pad_mask.device)

    # --- Start token (BOS) on device
    generated = torch.tensor([[bos_id]], dtype=torch.long, device=device)  # (1,1) on device

    for step in range(max_len):
        # Ensure generated on device
        generated = generated.to(device)
        tgt_pad_mask = (generated == pad_id).to(device)

        L = generated.size(1)
        # causal mask on device, float with -inf for masked positions
        tgt_mask = torch.triu(torch.ones((L, L), device=device) == 1, diagonal=1)
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))

        if DEBUG_DEVICES:
            print(f"DEBUG step {step}: generated device {generated.device}, tgt_mask device {tgt_mask.device}")

        # Forward - all args on device
        logits = model(
            src,
            generated,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask
        )  # (B, T, V)

        # Pick next token (on device)
        next_token = logits[:, -1, :].argmax(dim=-1)  # shape (B,)
        next_token = next_token.to(device)

        # Append
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Stop if EOS predicted
        if next_token.item() == eos_id:
            break

    # Drop BOS and convert to list of ids
    out_ids = generated[0].tolist()[1:]
    # Decode with sentencepiece (expects list of ids)
    return sp.decode(out_ids)


# =====================
# Main
# =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="path to checkpoint .pt")
    parser.add_argument("--src", required=True, help="source sentence (EN or JP)")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.model", help="sentencepiece model path")
    parser.add_argument("--debug-devices", action="store_true", help="print device debug info")
    args = parser.parse_args()

    global DEBUG_DEVICES
    DEBUG_DEVICES = args.debug_devices

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # load tokenizer
    sp = load_tokenizer(args.tokenizer)
    print("Loaded tokenizer vocab size:", sp.get_piece_size())

    # load checkpoint
    ckpt = torch.load(args.model, map_location="cpu")  # load to CPU first (safe)
    cfg = ckpt.get("config", {})
    model_cfg = cfg.get("model", {})

    # build model and move to device
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
        pad_id=model_cfg.get("pad_id", 3),
        tie_embeddings=model_cfg.get("tie_embeddings", True)
    ).to(device)

    # load weights into model (map tensors to device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print("Model loaded and moved to device.")

    # encode source (CPU tensor)
    src_cpu = encode_src(sp, args.src, max_len=model_cfg.get("max_len", 80), eos_id=model_cfg.get("eos_id", 2))
    print("Encoded src length (tokens):", src_cpu.size(0))

    # decode (everything inside handles device)
    translation = greedy_decode(model, src_cpu, sp, model_cfg, device=device)

    print("\n=== Translation ===")
    print("Input:", args.src)
    print("Output:", translation)


if __name__ == "__main__":
    main()
