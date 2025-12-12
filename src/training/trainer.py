# src/training/trainer.py
import os
import time
import math
import yaml
import torch
import random
import argparse
import datetime
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from src.data.dataset import TranslationDataset
from src.data.collate import MTDataCollator
from src.data.samplers import TokenBatchSampler
from src.models.transformer import MTTransformer

# Simple label smoothing loss
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super().__init__()
        self.confidence = 1.0 - label_smoothing
        self.smoothing = label_smoothing
        self.tgt_vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index
        self.kl = torch.nn.KLDivLoss(reduction="sum")

    def forward(self, pred, target):
        # pred: (batch * seq_len, vocab) logits
        # target: (batch * seq_len,)
        with torch.no_grad():
            true_dist = pred.data.clone()
            true_dist.fill_(self.smoothing / (self.tgt_vocab_size - 1))
            mask = (target != self.ignore_index)
            # Create smoothed target distribution
            true_dist[mask, target[mask]] = self.confidence
            true_dist[~mask] = 0
        log_prob = torch.nn.functional.log_softmax(pred, dim=-1)
        loss = self.kl(log_prob, true_dist)
        return loss / mask.sum().clamp(min=1)

# Noam scheduler (wraps optimizer)
class NoamOpt:
    def __init__(self, optimizer, d_model, warmup):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.d_model = d_model

    def step(self):
        self._step += 1
        lr = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self.optimizer.step()
        return lr

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        step = self._step if step is None else step
        return (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup ** -1.5)) if step > 0 else 0.0

def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def build_dataloader(cfg, split="train"):
    data_cfg = cfg["dataloader"]
    # use processed_clean paths
    if split == "train":
        src = "data/processed_clean/train.en"
        tgt = "data/processed_clean/train.ja"
    elif split == "valid":
        src = "data/processed_clean/valid.en"
        tgt = "data/processed_clean/valid.ja"
    else:
        src = "data/processed_clean/test.en"
        tgt = "data/processed_clean/test.ja"

    ds = TranslationDataset(src, tgt, "tokenizer/tokenizer.model",
                            max_source_len=data_cfg["max_source_len"],
                            max_target_len=data_cfg["max_target_len"])
    sampler = TokenBatchSampler(ds, max_tokens=data_cfg["max_tokens_per_batch"], shuffle=(split=="train" and data_cfg.get("shuffle", True)))
    collator = MTDataCollator()
    loader = DataLoader(ds, batch_sampler=sampler, collate_fn=collator, num_workers=data_cfg.get("num_workers", 2))
    return loader, ds

def train_loop(cfg_path, model_cfg_path, resume_path=None, device="cuda"):
    cfg = load_yaml(cfg_path)
    model_cfg = load_yaml(model_cfg_path)

    set_seed(cfg["train"]["seed"])
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    train_loader, _ = build_dataloader(cfg, "train")
    valid_loader, _ = build_dataloader(cfg, "valid")

    mcfg = cfg["model"]
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
    ).to(device)

    # optimizer & scheduler
    optim = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"], betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=cfg["optimizer"].get("weight_decay", 0.0))
    scheduler = NoamOpt(optim, d_model=mcfg["d_model"], warmup=cfg["scheduler"]["warmup_steps"])

    scaler = torch.cuda.amp.GradScaler(enabled=cfg["grad"]["use_amp"])
    grad_accum_steps = cfg["grad"]["grad_accum_steps"]
    max_grad_norm = cfg["grad"]["max_grad_norm"]

    # loss function with label smoothing
    criterion = LabelSmoothingLoss(cfg["grad"]["label_smoothing"], mcfg["tgt_vocab_size"], ignore_index=-100)

    start_epoch = 0
    global_step = 0
    best_valid_loss = float("inf")

    # Resume if requested
    if resume_path:
        cp = torch.load(resume_path, map_location=device)
        model.load_state_dict(cp["model_state"])
        optim.load_state_dict(cp["optim_state"])
        if "scaler_state" in cp:
            scaler.load_state_dict(cp["scaler_state"])
        start_epoch = cp.get("epoch", 0)
        global_step = cp.get("global_step", 0)
        best_valid_loss = cp.get("best_valid_loss", float("inf"))
        print("Resumed from", resume_path)

    checkpoint_dir = Path(cfg["checkpoint"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def evaluate():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in valid_loader:
                src = batch["src"].to(device)
                tgt_inp = batch["tgt_inp"].to(device)
                tgt_out = batch["tgt_out"].to(device)

                # Build masks: PyTorch expects True for padding positions
                src_pad_mask = (src == mcfg["pad_id"])
                tgt_pad_mask = (tgt_inp == mcfg["pad_id"])

                # tgt causal mask
                tgt_seq_len = tgt_inp.size(1)
                tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1, diagonal=1)
                tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))

                with torch.cuda.amp.autocast(enabled=cfg["grad"]["use_amp"]):
                    logits = model(src, tgt_inp, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask)
                    # reshape
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = tgt_out.view(-1)
                    loss = criterion(logits_flat, labels_flat)

                # aggregate
                non_pad = (labels_flat != -100).sum().item()
                total_loss += loss.item() * max(1, non_pad)
                total_tokens += max(1, non_pad)

        model.train()
        return total_loss / max(1, total_tokens)

    # Training loop
    num_epochs = cfg["train"]["num_epochs"]
    log_every = cfg["logging"]["log_every_steps"]
    checkpoint_every = cfg["checkpoint"]["checkpoint_every_steps"]

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_tokens = 0
        for batch in train_loader:
            global_step += 1
            src = batch["src"].to(device)
            tgt_inp = batch["tgt_inp"].to(device)
            tgt_out = batch["tgt_out"].to(device)

            # masks
            src_pad_mask = (src == mcfg["pad_id"])
            tgt_pad_mask = (tgt_inp == mcfg["pad_id"])
            tgt_seq_len = tgt_inp.size(1)
            tgt_mask = torch.triu(torch.ones((tgt_seq_len, tgt_seq_len), device=device) == 1, diagonal=1)
            tgt_mask = tgt_mask.float().masked_fill(tgt_mask, float('-inf'))

            # Forward/Backward with AMP + Grad Accum
            try:
                with torch.cuda.amp.autocast(enabled=cfg["grad"]["use_amp"]):
                    logits = model(src, tgt_inp, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask, tgt_mask=tgt_mask)
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = tgt_out.view(-1)
                    loss = criterion(logits_flat, labels_flat) / grad_accum_steps

                scaler.scale(loss).backward()
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # OOM-safe fallback: clear cache and attempt to split batch in halves and process sequentially
                    torch.cuda.empty_cache()
                    print(f"[OOM] Caught OOM at global_step {global_step}. Attempting fallback by splitting batch.")
                    # split batch into two smaller sub-batches and process them sequentially
                    try:
                        B = src.size(0)
                        mid = B // 2 or 1
                        indices = [list(range(0, mid)), list(range(mid, B))]
                        for idxs in indices:
                            if not idxs:
                                continue
                            src_sub = src[idxs].contiguous()
                            tgt_inp_sub = tgt_inp[idxs].contiguous()
                            tgt_out_sub = tgt_out[idxs].contiguous()
                            src_pad_mask_sub = (src_sub == mcfg["pad_id"])
                            tgt_pad_mask_sub = (tgt_inp_sub == mcfg["pad_id"])
                            tgt_seq_len_sub = tgt_inp_sub.size(1)
                            tgt_mask_sub = torch.triu(torch.ones((tgt_seq_len_sub, tgt_seq_len_sub), device=device) == 1, diagonal=1)
                            tgt_mask_sub = tgt_mask_sub.float().masked_fill(tgt_mask_sub, float('-inf'))

                            with torch.cuda.amp.autocast(enabled=cfg["grad"]["use_amp"]):
                                logits_sub = model(src_sub, tgt_inp_sub, src_key_padding_mask=src_pad_mask_sub, tgt_key_padding_mask=tgt_pad_mask_sub, tgt_mask=tgt_mask_sub)
                                logits_flat_sub = logits_sub.view(-1, logits_sub.size(-1))
                                labels_flat_sub = tgt_out_sub.view(-1)
                                loss_sub = criterion(logits_flat_sub, labels_flat_sub) / grad_accum_steps
                            scaler.scale(loss_sub).backward()
                    except RuntimeError as e2:
                        print("Fallback also failed. Skipping batch. Error:", e2)
                        torch.cuda.empty_cache()
                        scaler.update()
                        continue
                else:
                    raise e

            # when to update optimizer
            if global_step % grad_accum_steps == 0:
                # unscale, clip, step
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optim)
                scaler.update()
                scheduler.zero_grad()
                # scheduler.step() does optimizer.step() in our NoamOpt wrapper
                scheduler.step()
                optim.zero_grad()

            # logging
            if global_step % log_every == 0:
                # compute a forward-only loss value to report (no extra backward)
                print(f"[{datetime.datetime.now()}] Epoch {epoch} Step {global_step} LogLoss {loss.item()*grad_accum_steps:.4f}")

            if global_step % checkpoint_every == 0:
                cp_path = checkpoint_dir / f"ckpt_step_{global_step}.pt"
                state = {
                    "model_state": model.state_dict(),
                    "optim_state": optim.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "best_valid_loss": best_valid_loss,
                    "config": cfg
                }
                save_checkpoint(state, str(cp_path))
                print("Saved checkpoint:", cp_path)

        # epoch end: do validation
        valid_loss = evaluate()
        print(f"Epoch {epoch} finished in {time.time()-epoch_start:.1f}s valid_loss={valid_loss:.6f}")
        # save best
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_path = checkpoint_dir / "best.pt"
            state = {
                "model_state": model.state_dict(),
                "optim_state": optim.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_valid_loss": best_valid_loss,
                "config": cfg
            }
            save_checkpoint(state, str(best_path))
            print("Saved new best:", best_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

    train_loop(
        cfg_path=args.config,
        model_cfg_path=args.model_config,
        resume_path=args.resume,
        device=args.device
    )
