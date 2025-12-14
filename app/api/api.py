from fastapi import FastAPI
from pydantic import BaseModel
import torch
import sentencepiece as spm
from src.models.transformer import MTTransformer

app = FastAPI(title="MT Translation API")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load tokenizer ---
sp = spm.SentencePieceProcessor()
sp.load("tokenizer/tokenizer.model")

# --- Load model checkpoint ---
ckpt = torch.load("models/checkpoints/best.pt", map_location=DEVICE)
mcfg = ckpt["config"]["model"]

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
    tie_embeddings=mcfg["tie_embeddings"],
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
model.eval()

BOS = 1
EOS = 2
PAD = mcfg["pad_id"]

class Req(BaseModel):
    text: str

def translate(text):
    ids = sp.encode(text, out_type=int)
    ids = ids[:mcfg["max_len"] - 1] + [EOS]

    src = torch.tensor(ids).unsqueeze(0).to(DEVICE)
    src_mask = (src == PAD)

    with torch.no_grad():
        memory = model.encode(src, src_mask)

    out_ids = [BOS]
    for _ in range(mcfg["max_len"]):
        ys = torch.tensor([out_ids]).to(DEVICE)

        tgt_mask = torch.triu(
            torch.ones((ys.size(1), ys.size(1)), device=DEVICE) == 1, diagonal=1
        )
        tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float("-inf"))

        logits = model.decode(ys, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_mask)
        logits = model.generator(logits[:, -1])
        next_id = torch.argmax(logits, -1).item()

        if next_id == EOS:
            break
        out_ids.append(next_id)

    return sp.decode(out_ids[1:])

@app.post("/translate")
def _t(req: Req):
    return {"translation": translate(req.text)}
