# models/superglue.py

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


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim, layer_names):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim) for _ in layer_names])

    def forward(self, desc0, desc1):
        for layer in self.layers:
            delta0, delta1 = layer(desc0, desc1)
            desc0 = desc0 + delta0
            desc1 = desc1 + delta1
        return desc0, desc1


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(feature_dim, num_heads=4, batch_first=True)
        self.mlp = MLP(feature_dim * 2, feature_dim, 256)

    def forward(self, x, y):
        # Self-attention
        msg, _ = self.attn(x, y, y)
        return self.mlp(torch.cat([x, msg], dim=-1)), self.mlp(torch.cat([y, msg], dim=-1))


class SuperGlueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.descriptor_dim = 256
        self.gnn = AttentionalGNN(self.descriptor_dim, config.get("layers", ["self", "cross"] * 9))
        self.final_proj = nn.Linear(self.descriptor_dim, self.descriptor_dim)

    def forward(self, data):
        desc0 = data['descriptors0']  # [B, D, N]
        desc1 = data['descriptors1']

        # Convert to [B, N, D] for MultiheadAttention
        desc0 = desc0.transpose(1, 2)
        desc1 = desc1.transpose(1, 2)

        # Normalize
        desc0 = F.normalize(desc0, p=2, dim=-1)
        desc1 = F.normalize(desc1, p=2, dim=-1)

        desc0, desc1 = self.gnn(desc0, desc1)

        # Score computation
        scores = torch.einsum("bdn,bdm->bnm", desc0.transpose(1, 2), desc1.transpose(1, 2))
        matches0 = scores.argmax(dim=-1)
        return {"matches0": matches0}


def SuperGlue(config):
    model = SuperGlueNet(config)
    weights = torch.load(config["weights_path"], map_location="cpu")
    model.load_state_dict(weights)
    return model
