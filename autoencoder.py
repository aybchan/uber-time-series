import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(97,128),
            'lstm2': nn.LSTM(128,32),
        })
        
    
    def forward(self, x):
        x, _ = self.model['lstm1'](x)
        x = torch.tanh(x)
        x, _ = self.model['lstm2'](x)
        x = torch.tanh(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(32,128),
            'lstm2': nn.LSTM(128,97),
        })
    
    def forward(self, x):
        x, _ = self.model['lstm1'](x)
        x = torch.tanh(x)
        x, _ = self.model['lstm2'](x)
        return x


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.model = nn.Sequential(
            Encoder(),
            Decoder()
        )
    
    def forward(self, x):
        return self.model(x)