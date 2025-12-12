from src.data.dataset import TranslationDataset
from src.data.collate import MTDataCollator
from src.data.samplers import TokenBatchSampler
from torch.utils.data import DataLoader

dataset = TranslationDataset(
    "data/processed_clean/train.en",
    "data/processed_clean/train.ja",
    "tokenizer/tokenizer.model"
)

sampler = TokenBatchSampler(dataset, max_tokens=2500, shuffle=False)
collator = MTDataCollator()

loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collator)

for batch in loader:
    print("SRC SHAPE:", batch["src"].shape)
    print("TGT INP SHAPE:", batch["tgt_inp"].shape)
    print("TGT OUT SHAPE:", batch["tgt_out"].shape)
    break
