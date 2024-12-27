from torch import nn
import torch
from layers import Encode_Block_FC, Decode_Block_FC
from utils import ModelOutput


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        
        self.encode_strides = nn.ModuleList([])
        self.encode_stride_outputs = nn.ModuleList([])
        
        for i, stride in enumerate(args.n_encode_strides):
            encode_stride = nn.ModuleList([])
            for j in range(args.n_blocks_prior_encode_stride[i]):
                encode_stride.extend([
                    Encode_Block_FC(
                        args.hidden_dim_per_encode_block[i], 
                        args.n_residual_blocks_per_encode_block[i], 
                        args.hidden_dim_per_encode_block[i + 1],
                        args.n_residual_layers_per_block[i],
                        False,
                    ),
                ])
                
            self.encode_strides.extend([encode_stride])
            
            self.encode_stride_outputs.extend([
                Encode_Block_FC(
                    args.hidden_dim_per_encode_block[i], 
                    args.n_residual_blocks_per_encode_block[i], 
                    args.hidden_dim_per_encode_block[i + 1],
                    args.n_residual_layers_per_block[i],
                    stride,
                ),
            ])
            
        self.input_layer = nn.Linear(args.input_dim, args.hidden_dim_per_encode_block[0], bias=False)
        self.register_buffer('position_ids', torch.arange(args.input_dim) / args.input_dim)
        
    def forward(self, x):
        # x = self.input_layer(x + self.position_ids)
        x = self.input_layer(x)
        
        activations = []
        for encode_stride, encode_stride_output in zip(self.encode_strides, self.encode_stride_outputs):
            for encode_block in encode_stride:
                x, _ = encode_block(x)
                
            x, activation = encode_stride_output(x)
            activations.append(activation)
        
        return activations[::-1]
    
        
class Decoder(nn.Module):
    def __init__(self, args, hidden_dim_per_encode_block):
        super(Decoder, self).__init__()
        
        self.trainable_h = torch.nn.Parameter(data=torch.empty(size=(1, args.hidden_dim_per_decode_block[0])), requires_grad=True)                                 
        nn.init.kaiming_uniform_(self.trainable_h, nonlinearity='linear')
        
        self.decode_strides = nn.ModuleList([])
        self.decode_stride_outputs = nn.ModuleList([])
        
        for i, stride in enumerate(args.n_decode_strides):
            self.decode_stride_outputs.extend([
                Decode_Block_FC(
                    args.hidden_dim_per_decode_block[i - 1 if stride and i != 0 else i], 
                    args.latent_dim_per_decode_block[i], 
                    hidden_dim_per_encode_block[-2::-1][i], 
                    args.n_residual_blocks_per_decode_block[i],
                    args.hidden_dim_per_decode_block[i], 
                    args.n_residual_layers_per_block[i],
                    stride,
                ) 
            ])
            
            decode_stride = nn.ModuleList([])
            for j in range(args.n_blocks_after_decode_stride[i]):
                decode_stride.extend([
                    Decode_Block_FC(
                        args.hidden_dim_per_decode_block[i], 
                        args.latent_dim_per_decode_block[i], 
                        hidden_dim_per_encode_block[-2::-1][i], 
                        args.n_residual_blocks_per_decode_block[i],
                        args.hidden_dim_per_decode_block[i + 1], 
                        args.n_residual_layers_per_block[i],
                        False,
                    ),
                ])
                
            self.decode_strides.extend([decode_stride])
        
        self.output_layer = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(args.hidden_dim_per_decode_block[-1], args.output_dim),
        )
        
    def forward(self, activations):
        y = torch.tile(self.trainable_h, (activations[0].size()[0], 1,))
        posterior_dist_list = []
        prior_kl_dist_list = []
        
        for decode_stride, decode_stride_output, activation in zip(self.decode_strides, self.decode_stride_outputs, activations):
            y, posterior_dist, prior_kl_dist = decode_stride_output(activation, y)
            
            stride_posterior_dist = [posterior_dist]
            stride_prior_kl_dist = [prior_kl_dist]
            
            for decode_block in decode_stride:
                y, posterior_dist, prior_kl_dist = decode_block(activation, y)
                
                stride_posterior_dist.append(posterior_dist)
                stride_prior_kl_dist.append(prior_kl_dist)
                
            posterior_dist_list += stride_posterior_dist
            prior_kl_dist_list += stride_prior_kl_dist
            
        y = self.output_layer(y)
        
        return y, posterior_dist_list, prior_kl_dist_list
        
        
class MissVDVAE(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super(MissVDVAE, self).__init__()
        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config, encoder_config.hidden_dim_per_encode_block)
        
    def forward(self, x):
        activations = self.encoder(x)
        outputs, posterior_dist_list, prior_kl_dist_list = self.decoder(activations)
        
        return outputs, posterior_dist_list, prior_kl_dist_list