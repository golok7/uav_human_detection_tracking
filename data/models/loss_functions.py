import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bbox_loss = nn.SmoothL1Loss()
        self.conf_loss = nn.BCELoss()

    def forward(self, pred_boxes, pred_confs, gt_boxes):
        """
        pred_boxes: [B, N, 4]
        pred_confs: [B, N]
        gt_boxes:   list of tensors, each [Gi, 4]
        """

        batch_size, N, _ = pred_boxes.shape
        total_bbox_loss = 0.0
        total_conf_loss = 0.0

        for b in range(batch_size):
            gt = gt_boxes[b]
            num_gt = min(len(gt), N)

            if num_gt > 0:
                # Bounding box loss
                total_bbox_loss += self.bbox_loss(
                    pred_boxes[b, :num_gt],
                    gt[:num_gt]
                )

                # Confidence: 1 for valid boxes
                total_conf_loss += self.conf_loss(
                    pred_confs[b, :num_gt],
                    torch.ones(num_gt, device=pred_confs.device)
                )

            if num_gt < N:
                # Remaining predictions should be confidence 0
                total_conf_loss += self.conf_loss(
                    pred_confs[b, num_gt:],
                    torch.zeros(N - num_gt, device=pred_confs.device)
                )

        return (total_bbox_loss + total_conf_loss) / batch_size
