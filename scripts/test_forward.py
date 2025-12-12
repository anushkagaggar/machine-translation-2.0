import torch
from src.models.transformer import MTTransformer
from src.data.dataset import TranslationDataset
from src.data.collate import MTDataCollator
from src.data.samplers import TokenBatchSampler
from torch.utils.data import DataLoader

print("Loading dataset...")

ds = TranslationDataset(
    "data/processed_clean/train.en",
    "data/processed_clean/train.ja",
    "tokenizer/tokenizer.model",
    max_source_len=80,
    max_target_len=80,
)

sampler = TokenBatchSampler(ds, max_tokens=500)
collator = MTDataCollator()
loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collator)

batch = next(iter(loader))
print("Batch loaded!")

src = batch["src"]
tgt_inp = batch["tgt_inp"]
tgt_out = batch["tgt_out"]

print("src shape:", src.shape)
print("tgt_inp shape:", tgt_inp.shape)
print("tgt_out shape:", tgt_out.shape)

pad_id = 3
src_pad = (src == pad_id)
tgt_pad = (tgt_inp == pad_id)

seq_len = tgt_inp.size(1)
causal = torch.triu(torch.ones((seq_len, seq_len)) == 1, diagonal=1)
causal = causal.float().masked_fill(causal, float('-inf'))

print("Loading model...")
m = MTTransformer().eval()

with torch.amp.autocast("cuda", enabled=True):
    logits = m(
        src,
        tgt_inp,
        src_key_padding_mask=src_pad,
        tgt_key_padding_mask=tgt_pad,
        tgt_mask=causal,
    )

print("Logits shape:", logits.shape)
print("Forward test completed successfully!")
