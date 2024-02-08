import torch
import pytorch_lightning as pl
from torchmetrics.regression import MeanSquaredError

import numpy as np

from src.metrics import AmplitudeRMSE, PeriodRMSE, MagnitudeRMSE, WindowAccuracy


class PLWrapper(pl.LightningModule):
    def __init__(self, args, config, model, loss, model_name, th=0.5):
        super().__init__()
        self.model_name = model_name
        self.config = config
        self.model = model
        self.args = args
        self.loss = loss
        self.th = th

        # FIX: Hard coded 240 that represents 6 second window from -0.5 to 5.5 with freq 0.025
        self.offset = 240
        
        # Log hyperparameters
        self.save_hyperparameters(ignore=['loss', 'model'])

        # Metrics to track during training and validation
        self.train_amp_rmse = AmplitudeRMSE(squared=False)
        self.train_per_rmse = PeriodRMSE(squared=False)
        self.train_mag_rmse = MagnitudeRMSE(squared=False)
        self.train_wdw_rmse = MeanSquaredError(squared=False)
        self.train_wdw_acc = WindowAccuracy(th=self.th)
        self.val_amp_rmse = AmplitudeRMSE(squared=False)
        self.val_per_rmse = PeriodRMSE(squared=False)
        self.val_mag_rmse = MagnitudeRMSE(squared=False)
        self.val_wdw_rmse = MeanSquaredError(squared=False)
        self.val_wdw_acc = WindowAccuracy(th=self.th)

        # Metrics to test the model in different datasets
        self.test_wdw_acc = WindowAccuracy(th=self.th)


    def _get_per_amp_item(self, yp):
        yp = yp.flatten()

        yp_s = yp.argmax()
        yp_st = yp[yp_s]
        while yp_st > self.th:
            if yp_s > 0:
                yp_st = yp[yp_s]
                yp_s -= 1
            else:
                yp_st = self.th
        yp_s += 1

        yp_e = yp.argmax()
        yp_et = yp[yp_e]
        while yp_et > self.th:
            if yp_e < len(yp):
                yp_et = yp[yp_e]
                yp_e += 1
            else:
                yp_et = self.th
        yp_e -= 1

        return yp_s, yp_e


    def _get_per_amp(self, preds):
        wdw = []
        for yp in preds:
            yp_s, yp_e = self._get_per_amp_item(yp)
            wdw.append([yp_s, yp_e])
        wdw = torch.Tensor(wdw)
        return wdw
    
    def _transform_labels_binary(self, labels, t, freq):
        labels_pdf = []
        for i, l in enumerate(labels):
            left, right = l[0].item(), l[1].item()
            if right < left:
                left, right = right, left
            left = np.clip(left, a_min=self.args.start_time, a_max=self.args.end_time)
            right = np.clip(right, a_min=self.args.start_time, a_max=self.args.end_time)
            labels_pdf.append(list(np.zeros(max(0, -1*round(min(t)//freq)))) + 
                    list(np.zeros(round(left/freq))) + 
                    list(np.ones(round(right/freq) - round(left/freq) + 1)) + 
                    list(np.zeros(round(max(t)/freq)-round(right/freq))))

        labels = np.array(labels_pdf)
        labels = np.reshape(labels, tuple([s for s in labels.shape]+[1]))

        return labels


    def training_step(self, batch):
        # Training of the model on the training set
        self.model.train()
        x, y = batch
        x = x.float().permute(0, 2, 1)

        if self.config['metric'] == 'pdf':
            y = y.float().squeeze()
            y_hat = self.model(x).float()
        elif self.config['metric'] == 'None':
            y_orig = y.float().squeeze()
            y_hat_orig = self.model(x).float()
        
        # Make sure we always have 0.5 padding in the 5sec window
        st_wdw = (x.shape[2] - self.offset)//2

        if self.config['metric'] == 'None':
            # Build window based on limits
            t = np.arange(0, self.args.end_time-self.args.start_time+self.args.freq, self.args.freq)
            y = self._transform_labels_binary(y_orig, t, self.args.freq) 
            y = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                y, 
                                np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y_hat = self._transform_labels_binary(y_hat_orig, t, self.args.freq) 
            y_hat = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                    y_hat, 
                                    np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y, y_hat = y.squeeze(), y_hat.squeeze()
            y_hat = torch.from_numpy(y_hat)
            y = torch.from_numpy(y)

        y = y[:, st_wdw:st_wdw+self.offset]
        y_hat = y_hat[:, st_wdw:st_wdw+self.offset]

        y_wdw = self._get_per_amp(y)
        y_wdw_hat = self._get_per_amp(y_hat)
        
        if self.loss.__class__.__name__ == 'BCELoss':
            y_hat = torch.sigmoid(y_hat)
            train_loss = self.loss(y_hat, y)
        elif self.loss.__class__.__name__ == 'MSELoss':
            if self.config['metric'] == 'pdf':
                y_hat = torch.tanh(y_hat)
                train_loss = self.loss(y_hat, y)
            elif self.config['metric'] == 'None':
                train_loss = self.loss(y_hat_orig, y_orig)
        else:
            y_hat = torch.sigmoid(y_hat)
            train_loss = self.loss(x[:, 0:1, :], y_hat, y)

        # Metrics
        self.train_amp_rmse(x[:, 0:1, :], y_wdw_hat, y_wdw)
        self.train_per_rmse(y_wdw_hat, y_wdw)
        self.train_mag_rmse(x[:, 0:1, :], y_wdw_hat, y_wdw)
        self.train_wdw_rmse(y_hat, y)
        if self.config['metric'] == 'pdf':
            self.train_wdw_acc(y_wdw_hat, y_wdw)
        elif self.config['metric'] == 'None':
            self.train_wdw_acc(y_hat_orig*int(1/self.args.freq), y_orig*int(1/self.args.freq))
        
        # Logging the metrics
        self.log('train/loss', train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train/amp_rmse', self.train_amp_rmse, on_step=False, on_epoch=True)
        self.log('train/per_rmse', self.train_per_rmse, on_step=False, on_epoch=True)
        self.log('train/wdw_rmse', self.train_wdw_rmse, on_step=False, on_epoch=True)
        self.log('train/wdw_acc', self.train_wdw_acc, on_step=False, on_epoch=True, prog_bar=True)

        return train_loss


    def validation_step(self, batch):
        # Evaluation of the model on the validation set
        self.model.eval()
        x, y = batch
        x = x.float().permute(0, 2, 1)

        if self.config['metric'] == 'pdf':
            y = y.float().squeeze()
            y_hat = self.model(x).float()
        elif self.config['metric'] == 'None':
            y_orig = y.float().squeeze()
            y_hat_orig = self.model(x).float()
        
        # Make sure we always have 0.5 padding in the 5sec window
        st_wdw = (x.shape[2] - self.offset)//2

        if self.config['metric'] == 'None':
            # Build window based on limits
            t = np.arange(0, self.args.end_time-self.args.start_time+self.args.freq, self.args.freq)
            y = self._transform_labels_binary(y_orig, t, self.args.freq) 
            y = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                y, 
                                np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y_hat = self._transform_labels_binary(y_hat_orig, t, self.args.freq) 
            y_hat = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                    y_hat, 
                                    np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y, y_hat = y.squeeze(), y_hat.squeeze()
            y_hat = torch.from_numpy(y_hat)
            y = torch.from_numpy(y)

        y = y[:, st_wdw:st_wdw+self.offset]
        y_hat = y_hat[:, st_wdw:st_wdw+self.offset]

        y_wdw = self._get_per_amp(y)
        y_wdw_hat = self._get_per_amp(y_hat)
        
        if self.loss.__class__.__name__ == 'BCELoss':
            y_hat = torch.sigmoid(y_hat)
            val_loss = self.loss(y_hat, y)
        elif self.loss.__class__.__name__ == 'MSELoss':
            if self.config['metric'] == 'pdf':
                y_hat = torch.tanh(y_hat)
                val_loss = self.loss(y_hat, y)
            elif self.config['metric'] == 'None':
                val_loss = self.loss(y_hat_orig, y_orig)
        else:
            y_hat = torch.sigmoid(y_hat)
            val_loss = self.loss(x[:, 0:1, :], y_hat, y)

        # Metrics
        self.val_amp_rmse(x[:, 0:1, :], y_wdw_hat, y_wdw)
        self.val_per_rmse(y_wdw_hat, y_wdw)
        self.val_mag_rmse(x[:, 0:1, :], y_wdw_hat, y_wdw)
        self.val_wdw_rmse(y_hat, y)
        if self.config['metric'] == 'pdf':
            self.val_wdw_acc(y_wdw_hat, y_wdw)
        elif self.config['metric'] == 'None':
            self.val_wdw_acc(y_hat_orig*int(1/self.args.freq), y_orig*int(1/self.args.freq))
        
        # Logging the metrics
        self.log('val/loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/amp_rmse', self.val_amp_rmse, on_step=False, on_epoch=True)
        self.log('val/per_rmse', self.val_per_rmse, on_step=False, on_epoch=True)
        self.log('val/wdw_rmse', self.val_wdw_rmse, on_step=False, on_epoch=True)
        self.log('val/wdw_acc', self.val_wdw_acc, on_step=False, on_epoch=True, prog_bar=True)
        

    def predict_step(self, batch):
        # Prediction of the model on new data
        self.model.eval()
        x, y = batch
        x = x.float().permute(0, 2, 1)

        y = y.float().squeeze()
        y_hat = self.model(x)

        if self.config['metric'] == 'pdf':
            y_hat = torch.sigmoid(y_hat)
            y_hat = (y_hat - torch.min(y_hat, dim=1)[0][:, None]) / (torch.max(y_hat, dim=1)[0] - torch.min(y_hat, dim=1)[0])[:, None]
            y_hat = torch.clip(y_hat, min=0, max=1)
            # print(torch.min(y_hat), torch.max(y_hat))
        elif self.config['metric'] == 'None':
            # Build window based on limits
            t = np.arange(0, self.args.end_time-self.args.start_time+self.args.freq, self.args.freq)
            y = self._transform_labels_binary(y, t, self.args.freq) 
            y = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                y, 
                                np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y_hat = self._transform_labels_binary(y_hat, t, self.args.freq) 
            y_hat = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                    y_hat, 
                                    np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y, y_hat = y.squeeze(), y_hat.squeeze()
            y_hat = torch.from_numpy(y_hat)
            y = torch.from_numpy(y)
        
        assert(torch.min(y_hat).item() == 0)
        assert(torch.max(y_hat).item() == 1)

        # Make sure we always have 0.5 padding in the 5sec window
        st_wdw = (y.shape[1] - self.offset)//2
        y = y[:, st_wdw:st_wdw+self.offset]
        y_hat = y_hat[:, st_wdw:st_wdw+self.offset]

        y_wdw = self._get_per_amp(y)
        y_wdw_hat = self._get_per_amp(y_hat)

        return {'x': x, 'y': y, 'y_hat': y_hat, 'y_wdw': y_wdw, 'y_wdw_hat': y_wdw_hat}
    
    def test_step(self, batch):
        x, y = batch
        x = x.float().permute(0, 2, 1)

        y = y.float().squeeze()
        y_hat = self.model(x)

        if self.config['metric'] == 'pdf':
            y_hat = torch.sigmoid(y_hat)
            y_hat = (y_hat - torch.min(y_hat, dim=1)[0][:, None]) / (torch.max(y_hat, dim=1)[0] - torch.min(y_hat, dim=1)[0])[:, None]
            y_hat = torch.clip(y_hat, min=0, max=1)
        elif self.config['metric'] == 'None':
            # Build window based on limits
            t = np.arange(0, self.args.end_time-self.args.start_time+self.args.freq, self.args.freq)
            y = self._transform_labels_binary(y, t, self.args.freq) 
            y = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                y, 
                                np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y_hat = self._transform_labels_binary(y_hat, t, self.args.freq) 
            y_hat = np.concatenate([np.zeros((x.shape[0], round((self.args.padding)/self.args.freq), 1)), 
                                    y_hat, 
                                    np.zeros((x.shape[0], -1 + round((self.args.padding)/self.args.freq), 1))], axis=1)
            y, y_hat = y.squeeze(), y_hat.squeeze()
            y_hat = torch.from_numpy(y_hat)
            y = torch.from_numpy(y)
        
        assert(torch.min(y_hat).item() == 0)
        assert(torch.max(y_hat).item() == 1)

        # Make sure we always have 0.5 padding in the 5sec window
        st_wdw = (y.shape[1] - self.offset)//2
        y = y[:, st_wdw:st_wdw+self.offset]
        y_hat = y_hat[:, st_wdw:st_wdw+self.offset]

        y_wdw = self._get_per_amp(y)
        y_wdw_hat = self._get_per_amp(y_hat)

        self.test_wdw_acc(y_wdw_hat, y_wdw)

    def configure_optimizers(self):
        # Definition of the optimizer and the learning rate scheduler
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"])
        return optimizer