import argparse
import torch
import sacrebleu
from pathlib import Path
import sentencepiece as spm
from tqdm import tqdm
import yaml

from src.models.transformer import MTTransformer


# ------------------------------
# Greedy decode (batch = 1 only)
# ------------------------------
def greedy_decode(model, src, pad_id, device, max_len=80):
    model.eval()

    # src: tensor of shape [src_len]
    src = src.unsqueeze(0).to(device)  # → [1, src_len]
    src_pad_mask = (src == pad_id)

    with torch.no_grad():
        memory = model.encode(src, src_pad_mask)

    # Start with BOS=1
    ys = torch.tensor([[1]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_pad_mask = (ys == pad_id)

        # Causal mask
        tgt_mask = torch.triu(
            torch.ones((ys.size(1), ys.size(1)), device=device) == 1,
            diagonal=1
        )
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float("-inf"))

        with torch.no_grad():
            out = model.decode(
                tgt_tokens=ys,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=src_pad_mask
            )

            logits = model.generator(out)  # [1, seq, vocab]
            next_token = logits[:, -1, :].argmax(-1)   # [1]

        # Force scalar
        next_token = int(next_token[0].item())

        # Append
        ys = torch.cat([ys, torch.tensor([[next_token]], device=device)], dim=1)

        if next_token == 2:  # EOS
            break

    return ys.squeeze(0).tolist()


# ------------------------------
# Load model & checkpoint
# ------------------------------
def load_model(checkpoint_path, model_cfg_path, device):
    mcfg = yaml.safe_load(open(model_cfg_path, "r", encoding="utf-8"))["model"]

    model = MTTransformer(
        src_vocab_size=mcfg["src_vocab_size"],
        tgt_vocab_size=mcfg["tgt_vocab_size"],
        d_model=mcfg["d_model"],
        nhead=mcfg["nhead"],
        num_encoder_layers=mcfg["encoder_layers"],
        num_decoder_layers=mcfg["decoder_layers"],
        dim_feedforward=mcfg["d_ff"],
        dropout=mcfg["dropout"],
        max_len=mcfg["max_len"],
        pad_id=mcfg["pad_id"],
        tie_embeddings=mcfg["tie_embeddings"]
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    model.to(device)
    model.eval()

    return model, mcfg


# ------------------------------
# Main evaluation
# ------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model, mcfg = load_model(args.model, args.model_config, device)
    pad_id = mcfg["pad_id"]

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/tokenizer.model")

    # Load references
    src_lines = Path("data/processed_clean/test.en").read_text(encoding="utf-8").splitlines()
    tgt_lines = Path("data/processed_clean/test.ja").read_text(encoding="utf-8").splitlines()

    src_lines = src_lines[:args.limit]
    tgt_lines = tgt_lines[:args.limit]

    print(f"Evaluating on {len(src_lines)} samples...")

    system_outputs = []
    references = []

    for src, ref in tqdm(zip(src_lines, tgt_lines), total=len(src_lines)):
        # Encode source
        ids = sp.encode(src, out_type=int)
        ids = ids[:mcfg["max_len"] - 1] + [2]  # EOS

        ids = torch.tensor(ids, dtype=torch.long)

        # Decode
        hyp_ids = greedy_decode(model, ids, pad_id, device)
        hyp_text = sp.decode(hyp_ids)

        system_outputs.append(hyp_text)
        references.append(ref)

    bleu = sacrebleu.corpus_bleu(system_outputs, [references])
    print("\n=== BLEU SCORE ===")
    print(bleu)


if __name__ == "__main__":
    main()
