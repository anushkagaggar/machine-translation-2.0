# scripts/evaluate_bleu.py
import argparse
import torch
import sacrebleu
import sentencepiece as spm
from pathlib import Path

from src.models.transformer import MTTransformer
from src.data.dataset import TranslationDataset


def greedy_decode(model, src, pad_id, device, max_len=80):
    model.eval()
    src = src.unsqueeze(0).to(device)

    src_pad_mask = (src == pad_id)

    # Encode step
    memory = model.encode(src, src_key_padding_mask=src_pad_mask)

    # Start with BOS token
    ys = torch.tensor([[1]], dtype=torch.long, device=device)

    for _ in range(max_len):
        tgt_pad_mask = (ys == pad_id)
        seq_len = ys.size(1)

        # Create causal mask
        tgt_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=device) == 1,
            diagonal=1
        )
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))

        # DECODE (FULL SIGNATURE)
        out = model.decode(
            ys,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )

        logits = model.generator(out)
        next_token = logits[:, -1].argmax(dim=-1)

        ys = torch.cat([ys, next_token.unsqueeze(0)], dim=1)

        if next_token.item() == 2:  # EOS
            break

    return ys[0].tolist()


def load_model(checkpoint_path, model_cfg_path, device):
    import yaml

    full_cfg = yaml.safe_load(open(model_cfg_path, "r"))

    # FIX: model config is inside "model" key
    if "model" in full_cfg:
        mcfg = full_cfg["model"]
    else:
        mcfg = full_cfg  # fallback if someone uses a flat file

    model = MTTransformer(
        src_vocab_size = mcfg["src_vocab_size"],
        tgt_vocab_size = mcfg["tgt_vocab_size"],
        d_model        = mcfg["d_model"],
        nhead          = mcfg["nhead"],
        num_encoder_layers = mcfg["encoder_layers"],
        num_decoder_layers = mcfg["decoder_layers"],
        dim_feedforward = mcfg["d_ff"],
        dropout        = mcfg["dropout"],
        max_len        = mcfg["max_len"],
        pad_id         = mcfg["pad_id"],
        tie_embeddings = mcfg["tie_embeddings"]
    )

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, mcfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--src-file", default="data/processed_clean/test.en")
    parser.add_argument("--ref-file", default="data/processed_clean/test.ja")
    parser.add_argument("--limit", type=int, default=200)  # evaluate small subset
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer/tokenizer.model")

    # Load model
    model, mcfg = load_model(args.model, args.model_config, device)
    pad_id = mcfg["pad_id"]

    # Load test data
    src_lines = Path(args.src_file).read_text(encoding="utf-8").splitlines()
    ref_lines = Path(args.ref_file).read_text(encoding="utf-8").splitlines()

    src_lines = src_lines[:args.limit]
    ref_lines = ref_lines[:args.limit]

    print(f"Evaluating on {len(src_lines)} samples...")

    preds = []

    for i, src in enumerate(src_lines):
        ids = sp.encode(src, out_type=int)
        ids = ids[:mcfg["max_len"] - 1] + [2]  # EOS
        ids = torch.tensor(ids, dtype=torch.long)

        hyp_ids = greedy_decode(model, ids, pad_id, device)
        text = sp.decode(hyp_ids)
        preds.append(text)

        if i % 20 == 0:
            print(f"[{i}] src={src}\n     pred={text}\n")

    bleu = sacrebleu.corpus_bleu(preds, [ref_lines])
    print("\n=== BLEU SCORE ===")
    print(bleu)


if __name__ == "__main__":
    main()
