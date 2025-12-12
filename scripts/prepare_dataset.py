import json
import random
import argparse
import unicodedata
from pathlib import Path


def normalize(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u3000", " ")  # full-width space
    return text.strip()


def extract_pair(obj: dict):
    """
    Extract English↔Japanese pair assuming fixed structure:
    messages[1] = English (user)
    messages[2] = Japanese (assistant)
    """
    try:
        msgs = obj.get("messages", [])
        if len(msgs) < 3:
            return None, None

        en = msgs[1].get("content", "")
        ja = msgs[2].get("content", "")

        return en, ja

    except Exception:
        return None, None


def prepare(raw_path: str, out_dir: str,
            train_frac=0.80, valid_frac=0.10,
            min_chars=2, max_chars=1200, seed=42):

    raw_file = Path(raw_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_pairs = []

    with raw_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            en, ja = extract_pair(obj)
            en, ja = normalize(en), normalize(ja)

            if len(en) < min_chars or len(ja) < min_chars:
                continue

            if len(en) > max_chars:
                en = en[:max_chars]
            if len(ja) > max_chars:
                ja = ja[:max_chars]

            all_pairs.append((en, ja))

    print(f"Total usable pairs: {len(all_pairs)}")

    random.seed(seed)
    random.shuffle(all_pairs)

    N = len(all_pairs)
    n_train = int(N * train_frac)
    n_valid = int(N * valid_frac)

    train = all_pairs[:n_train]
    valid = all_pairs[n_train:n_train + n_valid]
    test = all_pairs[n_train + n_valid:]

    def save_split(name, items):
        en_path = out / f"{name}.en"
        ja_path = out / f"{name}.ja"

        with en_path.open("w", encoding="utf-8") as fe, \
             ja_path.open("w", encoding="utf-8") as fj:
            for en, ja in items:
                fe.write(en + "\n")
                fj.write(ja + "\n")

        print(f"Saved {name}: {len(items)} samples")

    save_split("train", train)
    save_split("valid", valid)
    save_split("test", test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=str,
        default="data/raw/Synthetic-JP-EN-Translation-Dataset-Magpie-Nemotron-4-20k.jsonl")
    parser.add_argument("--out", type=str, default="data/processed")
    args = parser.parse_args()

    prepare(args.raw, args.out)
