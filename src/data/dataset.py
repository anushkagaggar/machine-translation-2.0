from pathlib import Path
import torch
import sentencepiece as spm

class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, src_path, tgt_path, spm_model_path,
                 max_source_len=80, max_target_len=80):
        self.src_lines = Path(src_path).read_text(encoding="utf-8").splitlines()
        self.tgt_lines = Path(tgt_path).read_text(encoding="utf-8").splitlines()

        assert len(self.src_lines) == len(self.tgt_lines), "Unaligned dataset!"

        self.sp = spm.SentencePieceProcessor()
        self.sp.load(spm_model_path)

        self.bos = 1
        self.eos = 2

        self.max_src = max_source_len
        self.max_tgt = max_target_len

    def __len__(self):
        return len(self.src_lines)

    def encode_src(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = ids[:self.max_src - 1] + [self.eos]
        return torch.tensor(ids, dtype=torch.long)

    def encode_tgt(self, text):
        ids = self.sp.encode(text, out_type=int)
        ids = ids[:self.max_tgt - 2]

        inp = [self.bos] + ids
        out = ids + [self.eos]

        return torch.tensor(inp, dtype=torch.long), torch.tensor(out, dtype=torch.long)

    def __getitem__(self, idx):
        src = self.src_lines[idx]
        tgt = self.tgt_lines[idx]

        src_ids = self.encode_src(src)
        tgt_inp_ids, tgt_out_ids = self.encode_tgt(tgt)

        return {
            "src": src_ids,
            "tgt_inp": tgt_inp_ids,
            "tgt_out": tgt_out_ids
        }


# Utility function for inference
def load_sentencepiece_tokenizer(path):
    sp = spm.SentencePieceProcessor()
    sp.load(path)
    return sp