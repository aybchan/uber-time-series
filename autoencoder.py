import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(95, 128),
            'lstm2': nn.LSTM(128, 64),
            'lstm3': nn.LSTM(64, 1),
        })

    def forward(self, x):
        x, _ = self.model['lstm1'](x)
        x, _ = self.model['lstm2'](x)
        x, _ = self.model['lstm3'](x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(1, 64),
            'lstm2': nn.LSTM(64, 128),
            'lstm3': nn.LSTM(128, 1),
        })

    def forward(self, x):
        x, _ = self.model['lstm1'](x)
        x, _ = self.model['lstm2'](x)
        x, _ = self.model['lstm3'](x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc1 = nn.Linear(60, 32)
        self.fc2 = nn.Linear(32, 12)

    def forward(self, x):
        out = self.encoder(x)
        out = torch.cat((out, x[:, 36:, 3:4]), dim=1)
        out = self.decoder(out)
        out = self.fc1(out.view(-1, 60))
        out = self.fc2(out)

        return out
