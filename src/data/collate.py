import torch

PAD = 3

def pad_sequence(sequences, batch_first=False, padding_value=PAD):
    return torch.nn.utils.rnn.pad_sequence(
        sequences, batch_first=batch_first, padding_value=padding_value
    )

class MTDataCollator:
    def __call__(self, batch):
        src = [x["src"] for x in batch]
        tgt_inp = [x["tgt_inp"] for x in batch]
        tgt_out = [x["tgt_out"] for x in batch]

        src = pad_sequence(src, batch_first=True)
        tgt_inp = pad_sequence(tgt_inp, batch_first=True)
        tgt_out = pad_sequence(tgt_out, batch_first=True)

        src_mask = (src != PAD).long()
        tgt_mask = (tgt_inp != PAD).long()

        return {
            "src": src,
            "src_mask": src_mask,
            "tgt_inp": tgt_inp,
            "tgt_out": tgt_out,
            "tgt_mask": tgt_mask,
        }
