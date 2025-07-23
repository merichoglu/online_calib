import torch
import cv2
import numpy as np


class SuperPointFrontend:
    def __init__(self, model, device="cuda", keypoint_threshold=0.05):
        self.model = model.eval().to(device)
        self.device = device
        self.threshold = float(keypoint_threshold)

    def run(self, image):
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Normalize to [0,1]
        image = image.astype(np.float32) / 255.0
        # Create tensor of shape [1,1,H,W]
        inp = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(self.device)

        # Run official SuperPoint (expects dict input)
        with torch.no_grad():
            pred = self.model({"image": inp})

        # Official SuperPoint returns:
        #   pred['keypoints']: list of length B, each is Tensor[N,2]
        #   pred['scores']:     list of length B, each is Tensor[N]
        #   pred['descriptors']:list of length B, each is Tensor[D,N]

        # Extract for batch=0
        kpts = pred["keypoints"][0].cpu().numpy()  # shape [N, 2]
        scores = pred["scores"][0].cpu().numpy()  # shape [N]
        # descriptors come as [D, N]; transpose to [N, D]
        descs = pred["descriptors"][0].cpu().numpy().T  # shape [N, D]
        keep = scores >= self.threshold
        return kpts[keep], descs[keep], scores[keep]