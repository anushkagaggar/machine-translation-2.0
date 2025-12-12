import json
from pathlib import Path
from tqdm import tqdm
import argparse


# -----------------------------
# Language detection utilities
# -----------------------------
def is_japanese(text: str):
    for ch in text:
        # Hiragana / Katakana / CJK Unified Ideographs
        if (
            "\u3040" <= ch <= "\u30ff" or
            "\u4e00" <= ch <= "\u9faf" or
            "\u3400" <= ch <= "\u4dbf"
        ):
            return True
    return False


def is_english(text: str):
    for ch in text:
        if ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            return True
    return False


def is_pure_en(text: str):
    """True if text contains English and contains NO Japanese."""
    return is_english(text) and not is_japanese(text)


def is_pure_ja(text: str):
    """True if text contains Japanese and contains NO English."""
    return is_japanese(text) and not is_english(text)


# -----------------------------
# Load & extract all usable pairs
# -----------------------------
def load_pairs(path):
    usable = []

    with Path(path).open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading JSONL"):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            msgs = obj.get("messages", [])
            if len(msgs) < 3:
                continue

            user = msgs[1].get("content", "").strip()
            assistant = msgs[2].get("content", "").strip()

            if not user or not assistant:
                continue

            u_en = is_pure_en(user)
            u_ja = is_pure_ja(user)
            a_en = is_pure_en(assistant)
            a_ja = is_pure_ja(assistant)

            # VALID TRANSLATION CASES:

            # Case 1 — user EN, assistant JA
            if u_en and a_ja:
                usable.append((user, assistant))
                continue

            # Case 2 — user JA, assistant EN → FLIP
            if u_ja and a_en:
                usable.append((assistant, user))
                continue

            # All other cases: mixed language, or non-pure → discard
            continue

    return usable


# -----------------------------
# Saving function
# -----------------------------
def save_split(name, pairs, out_dir):
    en_path = out_dir / f"{name}.en"
    ja_path = out_dir / f"{name}.ja"

    with en_path.open("w", encoding="utf-8") as fe, ja_path.open("w", encoding="utf-8") as fj:
        for en, ja in pairs:
            fe.write(en.replace("\n", " ").strip() + "\n")
            fj.write(ja.replace("\n", " ").strip() + "\n")

    print(f"Saved {name}: {len(pairs)} samples")


# -----------------------------
# Main
# -----------------------------
def main(raw_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Extracting clean EN↔JA pairs...")
    pairs = load_pairs(raw_path)
    print("Total clean, aligned pairs:", len(pairs))

    # Shuffle
    import random
    random.seed(42)
    random.shuffle(pairs)

    # 80:10:10 split
    n = len(pairs)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    train = pairs[:n_train]
    valid = pairs[n_train:n_train+n_valid]
    test = pairs[n_train+n_valid:]

    save_split("train", train, out_dir)
    save_split("valid", valid, out_dir)
    save_split("test", test, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Input JSONL")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    main(args.raw, args.out)
