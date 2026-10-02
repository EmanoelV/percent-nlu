import torch
import torch.nn as nn

from percent_nlu.featurize import N_FEATS, MAX_LEN
from percent_nlu.schema import ROLES, MAX_SLOTS


class PercentNLU(nn.Module):
    def __init__(self, vocab_size, buckets, n_intents, d=192, layers=4, heads=4, ff=512, dropout=0.1,
                 use_feats=True):
        super().__init__()
        self.word = nn.Embedding(vocab_size, d, padding_idx=0)
        self.ngram = nn.Embedding(buckets, d, padding_idx=0)  # média manual (EmbeddingBag falha no MPS)
        self.pos = nn.Embedding(MAX_LEN, d)
        self.use_feats = use_feats
        if use_feats:
            self.feat = nn.ModuleList([nn.Embedding(c + 1, d, padding_idx=0) for c in N_FEATS])
        self.slot_flag = nn.Embedding(2, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.intent_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, n_intents))
        self.role_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, len(ROLES)))

    def forward(self, ids, ngs, feats, isslot, slot_pos):
        B, L = ids.shape
        mask = ids != 0
        ngm = (ngs != 0).unsqueeze(-1).float()
        ng = (self.ngram(ngs) * ngm).sum(2) / ngm.sum(2).clamp(min=1)
        x = self.word(ids) + ng + self.pos(torch.arange(L, device=ids.device))[None] + self.slot_flag(isslot)
        if self.use_feats:
            for i, emb in enumerate(self.feat):
                x = x + emb(feats[..., i])
        h = self.norm(self.enc(x, src_key_padding_mask=~mask))
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
        intent_logits = self.intent_head(pooled)
        idx = slot_pos.clamp(min=0)
        slot_h = torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        role_logits = self.role_head(slot_h)  # B x MAX_SLOTS x R
        return intent_logits, role_logits


def collate(batch, device):
    L = max(1, max(len(b["ids"]) for b in batch))
    G = max(1, max((len(g) for b in batch for g in b["ngs"]), default=1))
    B = len(batch)
    ids = torch.zeros(B, L, dtype=torch.long)
    ngs = torch.zeros(B, L, G, dtype=torch.long)
    feats = torch.zeros(B, L, len(N_FEATS), dtype=torch.long)
    isslot = torch.zeros(B, L, dtype=torch.long)
    slot_pos = torch.full((B, MAX_SLOTS), -1, dtype=torch.long)
    role_y = torch.full((B, MAX_SLOTS), -100, dtype=torch.long)
    intent_y = torch.zeros(B, dtype=torch.long)
    for i, b in enumerate(batch):
        n = len(b["ids"])
        ids[i, :n] = torch.tensor(b["ids"])
        for j, g in enumerate(b["ngs"]):
            if g:
                ngs[i, j, :len(g)] = torch.tensor(g)
        if n:
            feats[i, :n] = torch.tensor(b["feats"])
        for k, p in enumerate(b["slot_pos"]):
            if p >= 0:
                isslot[i, p] = 1
        slot_pos[i] = torch.tensor(b["slot_pos"])
        role_y[i] = torch.tensor(b["role_y"])
        intent_y[i] = b["intent_y"]
    t = lambda x: x.to(device, non_blocking=True)
    return t(ids), t(ngs), t(feats), t(isslot), t(slot_pos), t(role_y), t(intent_y)
