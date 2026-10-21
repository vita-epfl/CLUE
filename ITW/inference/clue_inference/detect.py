"""Person detection: torchvision's Faster R-CNN (ResNet-50 FPN v2, COCO weights), the COCO "person" class only.

At COCO's defaults (score 0.5, overlap suppression at IoU 0.5) people who sit close together get extra boxes: a head, a
torso. In videos the tracker turns these into extra people (six-friends example: 11 tracks, 7.7 people per frame for 6),
so videos use score >= 0.7 and suppress person boxes above IoU 0.4 (6 tracks, 6.0 per frame). Single images keep the
defaults, which also find people in unusual poses (the thrown judoka: score 0.6, IoU 0.5 with his opponent), and remove
the extra boxes from the meshes instead (pipeline.drop_duplicates).
"""

from __future__ import annotations

import numpy as np
import torch
import torchvision

PERSON = 1                                         # COCO category id in torchvision's detection models


class PersonDetector:
    def __init__(self, device: torch.device | str = "cuda", *, min_height: int = 40) -> None:
        from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

        self.device = torch.device(device)
        self.min_height = min_height
        self.model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1, box_score_thresh=0.5)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def __call__(self, frames_rgb: list[np.ndarray], *, score: float = 0.5, overlap: float = 0.5, batch: int = 8) -> list[np.ndarray]:
        """One (N, 5) array per frame: xyxy box in pixels + score, people at least ``min_height`` px tall with at least
        ``score``; of two person boxes overlapping above IoU ``overlap``, the lower-scoring one is dropped."""
        out = []
        for start in range(0, len(frames_rgb), batch):
            images = [torch.as_tensor(f, device=self.device).permute(2, 0, 1).float() / 255.0 for f in frames_rgb[start:start + batch]]
            for det in self.model(images):
                boxes, scores = det["boxes"], det["scores"]
                keep = (det["labels"] == PERSON) & ((boxes[:, 3] - boxes[:, 1]) >= self.min_height) & (scores >= score)
                boxes, scores = boxes[keep], scores[keep]
                keep = torchvision.ops.nms(boxes, scores, overlap).sort().values
                out.append(torch.cat([boxes[keep], scores[keep, None]], 1).float().cpu().numpy())
        return out
