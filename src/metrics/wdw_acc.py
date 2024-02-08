import torch
from torchmetrics import Metric


class WindowAccuracy(Metric):
    def __init__(self, th=0.5, tol=4, relative=False):
        super().__init__()
        self.th = th
        self.tol = tol
        self.relative = relative
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds_wdw, target_wdw):
        assert preds_wdw.shape == target_wdw.shape

        diff = torch.abs(preds_wdw - target_wdw)
        diff = torch.sum(diff, dim=1)
        if self.relative:
            diff /= (target_wdw[1] - target_wdw[0])
        count = torch.sum(diff <= self.tol)

        self.correct += count
        self.total += target_wdw.shape[0]

    def compute(self):
        return self.correct.float() / self.total