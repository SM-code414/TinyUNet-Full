import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundaryLoss(nn.Module):
    """
    Simple boundary-aware loss using Laplacian edges.

    Encourages predicted boundaries to align
    with GT boundaries.
    """

    def __init__(self):
        super().__init__()

        kernel = torch.tensor(
            [[[0, 0, 0],
              [0, -1, 0],
              [0, 0, 0]],

             [[0, -1, 0],
              [-1, 6, -1],
              [0, -1, 0]],

             [[0, 0, 0],
              [0, -1, 0],
              [0, 0, 0]]],
            dtype=torch.float32
        )

        self.register_buffer(
            "kernel",
            kernel.unsqueeze(0).unsqueeze(0)
        )

    def extract_boundary(self, x):

        B, C, D, H, W = x.shape

        edges = []

        for c in range(C):

            ch = x[:, c:c+1]

            edge = F.conv3d(
                ch,
                self.kernel,
                padding=1
            )

            edges.append(torch.abs(edge))

        return torch.cat(edges, dim=1)

    def forward(self, pred, target):

        """
        pred:
            logits BEFORE argmax
            (B,C,D,H,W)

        target:
            integer labels
            (B,D,H,W)
        """

        num_classes = pred.shape[1]

        pred_prob = torch.softmax(pred, dim=1)

        target_oh = F.one_hot(
            target.long(),
            num_classes=num_classes
        ).permute(0, 4, 1, 2, 3).float()

        pred_edge = self.extract_boundary(pred_prob)
        gt_edge = self.extract_boundary(target_oh)

        return F.l1_loss(pred_edge, gt_edge)
