import torch
import torch.nn as nn
import torch.nn.functional as F


class SuperPointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            self.relu,
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            self.relu,
            nn.MaxPool2d(2, 2),
        )

        self.det_head = nn.Conv2d(128, 65, 1)  # 64 + 1 dustbin
        self.desc_head = nn.Conv2d(128, 256, 1)

    def forward(self, x):
        x = self.encoder(x)

        # Detector head
        det = self.det_head(x)
        prob = self.softmax_pixelwise(det)

        # Descriptor head
        desc = self.desc_head(x)
        desc = F.normalize(desc, p=2, dim=1)

        return {"prob": prob, "descriptors": desc}

    def softmax_pixelwise(self, logits):
        b, c, h, w = logits.shape
        logits = logits.view(b, c, -1)
        prob = F.softmax(logits, dim=1)
        return prob.view(b, c, h, w)


def SuperPoint(config):
    model = SuperPointNet()
    weights = torch.load(config["weights_path"], map_location="cpu")
    # load_state_dict(strict=False) will ignore mismatches
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"[SuperPoint] Warning: missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[SuperPoint] Warning: unexpected keys in state_dict: {unexpected}")
    return model
