import sentencepiece as spm
import yaml
from pathlib import Path
import argparse


def load_config(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_spm_command(cfg):
    t = cfg["tokenizer"]

    cmd = (
        f"--input={t['input_corpus']} "
        f"--model_prefix={t['model_prefix']} "
        f"--vocab_size={t['vocab_size']} "
        f"--model_type={t['model_type']} "
        f"--character_coverage={t['character_coverage']} "
        f"--input_sentence_size={t['input_sentence_size']} "
        f"--shuffle_input_sentence={str(t['shuffle_input_sentence']).lower()} "
        f"--unk_surface={t['unk_surface']} "
        f"--bos_id={t['bos_id']} "
        f"--eos_id={t['eos_id']} "
        f"--unk_id={t['unk_id']} "
        f"--pad_id={t['pad_id']}"
    )

    return cmd


def main(cfg_path):
    cfg = load_config(cfg_path)
    cmd = build_spm_command(cfg)

    print("Running SentencePiece with command:")
    print("spm.SentencePieceTrainer.Train(")
    print(f"    '{cmd}'")
    print(")")

    spm.SentencePieceTrainer.Train(cmd)

    print("\nTokenizer training complete.")
    prefix = cfg["tokenizer"]["model_prefix"]
    print("Saved files:")
    print(prefix + ".model")
    print(prefix + ".vocab")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to tokenizer.yaml")
    args = ap.parse_args()

    main(args.config)
