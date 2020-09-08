import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(in_features, 32),
            'lstm2': nn.LSTM(32, 8),
            'lstm3': nn.LSTM(8, out_features)
        })

    def forward(self, x):
        out, _ = self.model['lstm1'](x)
        out, _ = self.model['lstm2'](out)
        out, _ = self.model['lstm3'](out)

        return out


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = nn.ModuleDict({
            'lstm1': nn.LSTM(1, 1),
            'lstm2': nn.LSTM(1, 1),
            'lstm3': nn.LSTM(1, 1)
        })

    def forward(self, x):
        out, _ = self.model['lstm1'](x)
        out, _ = self.model['lstm2'](out)
        out, _ = self.model['lstm3'](out)

        return out


class EncoderDecoder(nn.Module):
    def __init__(self, in_features, output_size):
        super(EncoderDecoder, self).__init__()
        self.output_size = output_size # F in the paper
        self.in_features = in_features
        self.traffic_col = 4

        self.model = nn.ModuleDict({
            'encoder': Encoder(self.in_features, 1),
            'decoder': Decoder(),
            'fc1': nn.Linear(60, 32),
            'fc2': nn.Linear(32, self.output_size)
        })

    def forward(self, x):
        out = self.model['encoder'](x)

        # concatenate the auxiliary time series values
        # with the output from the encoder (the embedding)
        x_auxiliary = x[:,-self.output_size:,[self.traffic_col]]
        decoder_input = torch.cat([out, x_auxiliary], dim=1)

        out = self.model['decoder'](decoder_input)
        out = self.model['fc1'](out.view(-1, 60))
        out = self.model['fc2'](out)

        return out
