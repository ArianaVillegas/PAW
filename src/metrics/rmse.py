import torch
from torchmetrics import Metric
from torchmetrics.regression import MeanSquaredError


class AmplitudeRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def _get_amplitude_item(self, x, wdw):
        wdw = wdw.numpy().astype(int)
        if wdw[0] >= wdw[1]:
            return 0
        x_wdw = x[wdw[0]:wdw[1]]
        amp = (x_wdw.max() - x_wdw.min())/2
        return amp

    def _get_amplitude(self, x, wdw):
        amp = []
        for xi, wi in zip(x, wdw):
            x_amp = self._get_amplitude_item(xi.squeeze(), wi)
            amp.append(x_amp)
        amp = torch.Tensor(amp)
        return amp

    def update(self, x, pred_wdw, target_wdw):
        assert pred_wdw.shape == target_wdw.shape
        pred_amp = self._get_amplitude(x, pred_wdw)
        pred_amp = torch.where(pred_amp > 0, pred_amp, 0)
        target_amp = self._get_amplitude(x, target_wdw)
        self.rmse(pred_amp, target_amp)

    def compute(self):
        return self.rmse.compute()


class PeriodRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def update(self, pred_wdw, target_wdw, freq=0.025):
        assert pred_wdw.shape == target_wdw.shape
        pred_per = pred_wdw[:, 1] - pred_wdw[:, 0]
        pred_per = torch.where(pred_per > 0, pred_per, 0) * freq * 2
        target_per = (target_wdw[:, 1] - target_wdw[:, 0]) * freq * 2
        self.rmse(pred_per, target_per)

    def compute(self):
        return self.rmse.compute()


class MagnitudeRMSE(Metric):
    def __init__(self, squared, num_outputs=1):
        super().__init__()
        self.rmse = MeanSquaredError(squared=squared)

    def _get_amplitude_item(self, x, wdw):
        wdw = wdw.numpy().astype(int)
        if wdw[0] >= wdw[1]:
            return 0
        x_wdw = x[wdw[0]:wdw[1]]
        amp = (x_wdw.max() - x_wdw.min())/2
        return amp

    def _get_amplitude(self, x, wdw):
        amp = []
        for xi, wi in zip(x, wdw):
            x_amp = self._get_amplitude_item(xi.squeeze(), wi)
            amp.append(x_amp)
        amp = torch.Tensor(amp)
        return amp

    def update(self, x, pred_wdw, target_wdw, freq=0.025):
        assert pred_wdw.shape == target_wdw.shape
        pred_per = pred_wdw[:, 1] - pred_wdw[:, 0]
        pred_per = torch.where(pred_per > 0, pred_per, 0) * freq * 2
        target_per = (target_wdw[:, 1] - target_wdw[:, 0]) * freq * 2

        pred_amp = self._get_amplitude(x, pred_wdw)
        pred_amp = torch.where(pred_amp > 0, pred_amp, 0)
        target_amp = self._get_amplitude(x, target_wdw)

        pred_mag = torch.where((pred_amp == 0) | (pred_per == 0), 0, torch.log(pred_amp) - torch.log(pred_per))
        target_mag = torch.where((target_amp == 0) | (target_per == 0), 0, torch.log(target_amp) - torch.log(target_per))

        self.rmse(pred_mag, target_mag)

    def compute(self):
        return self.rmse.compute()