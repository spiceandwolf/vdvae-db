import torch.nn as nn


class ResBlock_FC(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, n_residual_layers_per_block = 0):
        super(ResBlock_FC, self).__init__()
        assert out_channels==in_channels 
        
        residual_layers = [
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Linear(in_channels, middle_channels, bias=False),  
        ]
        
        for _ in range(n_residual_layers_per_block):
            residual_layers.append(nn.ReLU())
            residual_layers.append(nn.Linear(middle_channels, middle_channels, bias=False))
            
        residual_layers += [
            nn.BatchNorm1d(middle_channels),
            nn.ReLU(),
            nn.Linear(middle_channels, out_channels, bias=False),  
        ]
        
        self.residual_layers = nn.Sequential(*residual_layers)
        
    def forward(self, x):
        inputs = x
        x = self.residual_layers(x)
        output = x + inputs
        return output