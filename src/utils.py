import os
import numpy as np

import torch
import torch.nn as nn

from sklearn.preprocessing import StandardScaler, RobustScaler

from src.data import transform_labels_gaussian, transform_labels_orig, transform_labels_binary


# helper functions
def create_path(file_path):
    if not os.path.exists(file_path):
        print(file_path.split('/')[1:-1], file_path)
        for i, folder in enumerate(file_path.split('/')[1:-1]):
            print(i)
            i += 1
            if not os.path.exists('/'.join(file_path.split('/')[:i] + [folder])):
                os.mkdir('/'.join(file_path.split('/')[:i] + [folder]))



def get_per_amp(yp, th=0.5):
    yp = yp.flatten()

    yp_s = yp.argmax()
    yp_st = yp[yp_s]
    while yp_st > th:
        if yp_s > 0:
            yp_st = yp[yp_s]
            yp_s -= 1
        else:
            yp_st = th
    yp_s += 1

    yp_e = yp.argmax()
    yp_et = yp[yp_e]
    while yp_et > th:
        if yp_e < len(yp):
            yp_et = yp[yp_e]
            yp_e += 1
        else:
            yp_et = th
    yp_e -= 1

    return yp_s, yp_e



def get_pred_tensor(x_true, y_true, y_pred, freq=0.025):
    amp_true, per_true = [], []
    amp_pred, per_pred = [], []
    for xt, yt, yp in zip(x_true, y_true, y_pred):
        xt = xt.squeeze()
        yt_nonzero = np.nonzero(yt).squeeze()
        if len(yt_nonzero) > 0:
            yt_s = yt_nonzero.min()
            yt_e = yt_nonzero.max()
            yt_per = yt_e - yt_s
            per_true.append(yt_per*freq)
            
            yp_s, yp_e = get_per_amp(yp)
            yp_per = yp_e - yp_s
            per_pred.append(yp_per*freq)

            mpeaks_t = np.array(xt[yt_s:yt_e], dtype=np.float64)
            mpeaks_t = np.abs(np.max(mpeaks_t) - np.min(mpeaks_t)) / 2
            amp_true.append(mpeaks_t)
            if yp_s < yp_e:
                mpeaks_p = np.array(xt[yp_s:yp_e], dtype=np.float64)
                mpeaks_p = np.abs(np.max(mpeaks_p) - np.min(mpeaks_p)) / 2
                amp_pred.append(mpeaks_p)
            else:
                amp_pred.append(0)
    per_true = torch.FloatTensor(per_true).requires_grad_(True)
    per_pred = torch.FloatTensor(per_pred).requires_grad_(True)
    amp_true = torch.FloatTensor(amp_true).requires_grad_(True)
    amp_pred = torch.FloatTensor(amp_pred).requires_grad_(True)
    return per_true, per_pred, amp_true, amp_pred



class AmpPerLoss(nn.Module):
    def __init__(self, weight=0.1, ratio=1):
        super().__init__()
        self.loss_amp = nn.MSELoss()
        self.loss_per = nn.MSELoss()
        self.loss_ae = nn.BCELoss()
        self.weight = weight
        self.ratio = ratio
    
    def forward(self, x_true, y_pred, y_true):
        per_true, per_pred, amp_true, amp_pred = get_pred_tensor(x_true, y_true, y_pred)
        per_loss = self.loss_per(per_pred, per_true) * self.weight / 5
        amp_loss = self.loss_amp(amp_pred, amp_true) * self.weight
        loss = torch.norm(torch.stack([per_loss, amp_loss]), p=2)
        ae_loss = self.loss_ae(y_pred, y_true)
        return loss + ae_loss


def preprocess(data_fil, labels_fil, metric, t, freq, scale_type, padding, label_type):
    if metric == 'pdf':
        if label_type == 'binary':
            labels_fil = transform_labels_binary(labels_fil, t, freq) 
        elif label_type == 'gaussian':
            labels_fil = transform_labels_gaussian(labels_fil, t, freq) 
        elif label_type == 'orig':
            labels_fil = transform_labels_orig(labels_fil, data_fil, t, freq) 
        labels_fil = np.concatenate([np.zeros((labels_fil.shape[0], round((padding)/freq), 1)), 
                                    labels_fil, 
                                    np.zeros((labels_fil.shape[0], -1 + round((padding)/freq), 1))], axis=1)
    if len(labels_fil) and scale_type != None:
        if scale_type == 'Standard' or scale_type == 'Robust':
            # Scaling
            if scale_type == 'Standard':
                scaler = StandardScaler()
            elif scale_type == 'Robust':
                scaler = RobustScaler()
            # Fit and transform
            data_fil = scaler.fit_transform(data_fil[:,:,0])
        elif scale_type == 'MinMax':
            max_fil = np.max(np.abs(data_fil[:,:,0]), axis=1)
            data_fil = np.divide(data_fil[:,:,0].T, max_fil).T
        # Reshape
        data_fil = np.reshape(data_fil, (data_fil.shape[0], data_fil.shape[1], 1))
    return data_fil, labels_fil