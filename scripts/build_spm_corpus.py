from pathlib import Path
import argparse

def main(train_en, train_ja, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        for path in [train_en, train_ja]:
            with Path(path).open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")

    print("Joint corpus built at:", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_en", required=True)
    ap.add_argument("--train_ja", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    main(args.train_en, args.train_ja, args.out)
