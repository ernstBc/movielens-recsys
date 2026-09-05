from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class AutoEncoder(nn.Module):
    def __init__(self, io_features:int, hidden_dim:int):
        super().__init__()

        self.encoder = nn.Linear(in_features=io_features, out_features=hidden_dim)
        self.decoder = nn.Linear(in_features=hidden_dim, out_features=io_features)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


    def forward(self, x):
        x = F.relu(self.encoder(x))
        x = self.decoder(x)

        return x



class DeepAutoEncoder(nn.Module):
    def __init__(self, io_features:int, hidden_dims:list[int], dropout_rate:float) -> None:
        super().__init__()

        encoder = [nn.Linear(in_features=io_features, out_features=hidden_dims[0]),
                                      nn.ReLU(), 
                                      nn.Dropout(dropout_rate)]
        for prev_dims, pos_dims in zip(hidden_dims[:-1], hidden_dims[1:]):
            encoder.append(nn.Linear(in_features=prev_dims, out_features=pos_dims))
            encoder.append(nn.ReLU())
            encoder.append(nn.Dropout(dropout_rate))

        decoder = []
        for prev_dims, pos_dims in zip(reversed(hidden_dims[1:]), reversed(hidden_dims[:-1])):
            decoder.append(nn.Linear(in_features=prev_dims, out_features=pos_dims))
            decoder.append(nn.ReLU())
            decoder.append(nn.Dropout(dropout_rate))
        decoder.append(nn.Linear(in_features=hidden_dims[0], out_features=io_features))

        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


if __name__=='__main__':
    n_items = 9000

    example_tensor = torch.rand(size=(32, n_items)) * 5.0

    autoencoder = AutoEncoder(n_items, 64)
    deep_ae = DeepAutoEncoder(n_items, [64,32,16,8], 0.6)

    ae_output = autoencoder(example_tensor)
    d_ae_output = deep_ae(example_tensor)


    print(ae_output.shape)
    print(d_ae_output.shape)