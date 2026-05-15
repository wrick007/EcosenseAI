import torch
import torch.nn as nn
import torch.nn.functional as F

IMG_W = 216


class TBlock(nn.Module):
    def __init__(self, d, h, mlp=4.0, drop=0.1):
        super().__init__()

        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(
            embed_dim=d,
            num_heads=h,
            dropout=drop,
            batch_first=True,
        )

        self.n2 = nn.LayerNorm(d)

        md = int(d * mlp)

        self.ff = nn.Sequential(
            nn.Linear(d, md),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(md, d),
            nn.Dropout(drop),
        )

    def forward(self, x):
        a, _ = self.attn(self.n1(x), self.n1(x), self.n1(x))
        x = x + a
        x = x + self.ff(self.n2(x))
        return x


class AudioViT(nn.Module):
    def __init__(self, H=128, W=216, p=16, d=384, depth=12, heads=6):
        super().__init__()

        self.p = p

        nh = H // p
        nw = W // p
        self.N = nh * nw

        self.embed = nn.Conv2d(1, d, kernel_size=p, stride=p)

        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, self.N + 1, d))

        self.blocks = nn.ModuleList(
            [TBlock(d, heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(d)

        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, mask_ratio=0.0):
        B = x.shape[0]

        if x.shape[-1] < IMG_W:
            x = F.pad(x, (0, IMG_W - x.shape[-1]))
        else:
            x = x[..., :IMG_W]

        x = self.embed(x).flatten(2).transpose(1, 2)
        x = x + self.pos[:, 1:, :]

        cls = self.cls + self.pos[:, :1, :]
        cls = cls.expand(B, -1, -1)

        x = torch.cat([cls, x], dim=1)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x, None, None


class EcologicalHead(nn.Module):
    def __init__(self, d=384, n_idx=4):
        super().__init__()

        def make_head():
            return nn.Sequential(
                nn.LayerNorm(d),
                nn.Linear(d, 128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, 1),
                nn.Sigmoid(),
            )

        self.species = make_head()
        self.anthro = make_head()
        self.geo = make_head()

        self.idx_proj = nn.Sequential(
            nn.Linear(n_idx, 32),
            nn.GELU(),
            nn.Linear(32, 32),
        )

        self.proj = nn.Sequential(
            nn.Linear(d + 3 + 32, 512),
            nn.GELU(),
            nn.LayerNorm(512),
        )

    def forward(self, cls, idx):
        sp = self.species(cls)
        an = self.anthro(cls)
        geo = self.geo(cls)

        idx_feat = self.idx_proj(idx)

        feat = torch.cat([cls, sp, an, geo, idx_feat], dim=-1)
        feat = self.proj(feat)

        return feat, {
            "species": sp,
            "anthro": an,
            "geo": geo,
        }


class TemporalLEI(nn.Module):
    def __init__(self, d=512, S=6, depth=2, heads=4):
        super().__init__()

        self.pos = nn.Parameter(torch.zeros(1, S, d))

        self.blks = nn.ModuleList(
            [TBlock(d, heads) for _ in range(depth)]
        )

        self.norm = nn.LayerNorm(d)

        self.attn_pool = nn.Linear(d, 1)

        self.lei = nn.Sequential(
            nn.Linear(d, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.rank = nn.Sequential(
            nn.Linear(d, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, seq):
        x = seq + self.pos[:, :seq.shape[1], :]

        for block in self.blks:
            x = block(x)

        x = self.norm(x)

        w = torch.softmax(self.attn_pool(x), dim=1)
        x = (x * w).sum(dim=1)

        lei = self.lei(x).squeeze(-1)
        rank = self.rank(x).squeeze(-1)

        return lei, rank


class EcoSenseModel(nn.Module):
    def __init__(self, d=384, seq_len=6, n_idx=4):
        super().__init__()

        self.encoder = AudioViT(d=d)
        self.eco = EcologicalHead(d=d, n_idx=n_idx)
        self.agg = TemporalLEI(d=512, S=seq_len)

    def features(self, x, idx):
        tokens, _, _ = self.encoder(x, mask_ratio=0.0)
        cls = tokens[:, 0, :]
        return self.eco(cls, idx)

    def forward(self, seq, idxs):
        B, S, C, H, W = seq.shape

        seq_flat = seq.view(B * S, C, H, W)
        idx_flat = idxs.view(B * S, -1)

        feat, scores = self.features(seq_flat, idx_flat)

        feat = feat.view(B, S, -1)

        lei, rank = self.agg(feat)

        return lei, rank, scores