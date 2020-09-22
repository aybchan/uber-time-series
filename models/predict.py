import torch
import torch.nn as nn

class Predict(nn.Module):
    def __init__(self, params, p, encoder_decoder: nn.Module):
        super(Predict, self).__init__()
        
        self.encoder = encoder_decoder.model['encoder'].eval()
        self.params = params
        self.model = nn.Sequential(
            nn.Linear(params['n_extracted_features'] + params['n_external_features'], params['predict_hidden_1']),
            nn.Dropout(p),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(params['predict_hidden_1'], params['predict_hidden_2']),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(params['predict_hidden_2'], params['n_output_steps'])
        )

    def forward(self, x):
        x_input, external = x
        extracted = self.encoder(x_input).view(-1, self.params['n_extracted_features'])
        x_concat = torch.cat([extracted, external], dim=-1)
        out = self.model(x_concat)
        return out