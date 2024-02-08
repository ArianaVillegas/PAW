import torch.nn as nn
import torch.nn.functional as F

    

class SpatialDropout1D(nn.Module):
    def __init__(self, dropout_rate):
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(2)
        x = self.dropout(x)
        x = x.squeeze(2) 
        return x


class CNN(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, drop_rate):
        super(CNN, self).__init__()
        self.batchnorm1d = nn.BatchNorm1d(in_filters)
        self.dropout = SpatialDropout1D(drop_rate)
        self.conv1d = nn.Conv1d(in_filters, out_filters, kernel_size, padding=kernel_size//2)

    def forward(self, x):
        x = self.batchnorm1d(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv1d(x)
        return x


class LSTM(nn.Module):
    def __init__(self, nb_filters):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(nb_filters, nb_filters, bidirectional=True, batch_first=True)
        self.conv1d = nn.Conv1d(nb_filters*2, nb_filters, kernel_size=1, stride=1)
        self.batchnorm1d = nn.BatchNorm1d(nb_filters)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x.transpose(1, 2)  
        x = self.conv1d(x)
        x = self.batchnorm1d(x)
        return x


class PAW(nn.Module):
    def __init__(self, config):
        super().__init__()
        config['drop_rate'] = 0.2
        config['n_cnn'] = 1
        config['n_lstm'] = 1
        config['n_transformer'] = 1
        config['nb_filters'] = [32, 64]
        config['kernel_size'] = [13, 13]
        self.config = config
        self.nb_filters = config['nb_filters']
        self.kernel_size = config['kernel_size']
        self.encoder_depth = len(config['nb_filters'])
        self.decoder_depth = len(config['nb_filters'])
        self.n_cnn = config['n_cnn']
        self.n_lstm = config['n_lstm']
        self.n_transformer = config['n_transformer']
        self.drop_rate = config['drop_rate']
        self.out_shape = config['out_shape']

        self.encoder = self._encoder()
        self.cnn_block = nn.ModuleList([CNN(self.nb_filters[-1], self.nb_filters[-1], self.kernel_size[-1], self.drop_rate) 
                                        for _ in range(self.n_cnn)])
        self.lstm_block = nn.ModuleList([LSTM(self.nb_filters[-1]) 
                                         for _ in range(self.n_lstm)])
        self.transformer_block = nn.ModuleList([nn.MultiheadAttention(self.nb_filters[-1], 4)
                                                for _ in range(self.n_transformer)])
        self.decoder = self._decoder()
        self.conv_out = nn.Conv1d(self.nb_filters[-1], 1, self.kernel_size[-1], padding=self.kernel_size[-1]//2)

    def forward(self, x):
        x = self.encoder(x)
        for cnn in self.cnn_block:
            x = cnn(x)
        for lstm in self.lstm_block:
            x = x.permute(0, 2, 1)
            x = lstm(x)
        x = x.permute(0, 2, 1)
        for transformer in self.transformer_block:
            x, _ = transformer(x, x, x)
        x = x.permute(0, 2, 1)
        x = self.decoder(x)
        x = self.conv_out(x)
        x = x.squeeze()
        return x

    def _encoder(self):
        layers = []
        in_channels = self.out_shape
        for dp in range(self.encoder_depth):
            layers.append(nn.Conv1d(in_channels, self.nb_filters[dp], self.kernel_size[dp], padding=self.kernel_size[dp]//2))
            layers.append(nn.MaxPool1d(2))
            in_channels = self.nb_filters[dp]
        return nn.Sequential(*layers)

    def _decoder(self):
        layers = []
        in_channels = self.nb_filters[-1]
        for dp in range(self.decoder_depth):
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.Conv1d(in_channels, self.nb_filters[dp], self.kernel_size[dp], padding=self.kernel_size[dp]//2))
            in_channels = self.nb_filters[dp]
        return nn.Sequential(*layers)
        