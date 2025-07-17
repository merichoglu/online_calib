import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, ch_in, ch_out, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ch_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, ch_out),
        )

    def forward(self, x):
        return self.net(x)


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.mlp = MLP(feature_dim * 2, feature_dim, 256)

    def forward(self, x, y):
        # x attends to y→y, msg0 shape [B, len(x), D]
        msg0, _ = self.attn(x, y, y)
        # y attends to x→x, msg1 shape [B, len(y), D]
        msg1, _ = self.attn(y, x, x)

        # propagate
        delta0 = self.mlp(torch.cat([x, msg0], dim=-1))  # shape [B, len(x), D]
        delta1 = self.mlp(torch.cat([y, msg1], dim=-1))  # shape [B, len(y), D]
        return delta0, delta1


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList(
            [AttentionalPropagation(feature_dim) for _ in layer_names]
        )

    def forward(self, desc0, desc1):
        for layer in self.layers:
            delta0, delta1 = layer(desc0, desc1)
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1
        return desc0, desc1


class SuperGlueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.descriptor_dim = 256
        self.gnn = AttentionalGNN(
            self.descriptor_dim, config.get("layers", ["self", "cross"] * 9)
        )
        self.final_proj = nn.Linear(self.descriptor_dim, self.descriptor_dim)

    def forward(self, data):
        desc0 = data["descriptors0"]  # [B, D, N]
        desc1 = data["descriptors1"]  # [B, D, M]

        # swap to [B, N, D] and [B, M, D]
        desc0 = desc0.transpose(1, 2)
        desc1 = desc1.transpose(1, 2)

        # normalize
        desc0 = F.normalize(desc0, p=2, dim=-1)
        desc1 = F.normalize(desc1, p=2, dim=-1)

        # graph propagation
        desc0, desc1 = self.gnn(desc0, desc1)

        # matching scores [B, N, M]
        scores = torch.einsum("bnd,bmd->bnm", desc0, desc1)  # [B,N,D]  # [B,M,D]
        matches0 = scores.argmax(dim=-1)  # [B, N]
        return {"matches0": matches0}


def SuperGlue(config):
    model = SuperGlueNet(config)
    weights = torch.load(config["weights_path"], map_location="cpu")

    # strict=False to ignore missing/unexpected keys
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[SuperGlue] Warning: missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[SuperGlue] Warning: unexpected keys in state_dict: {unexpected}")
    return model
