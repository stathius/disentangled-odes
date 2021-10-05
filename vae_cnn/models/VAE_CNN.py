import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')

class VAE_CNN(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 fc_dim: int,
                 latent_dim: int,
                 hidden_dims: List,
                 filter_size: int,
                 padding: int,
                 stride: int,
                 nonlinearity: str,
                 dropout_pct: int,
                 use_layer_norm: bool
                 ) -> None:
        super().__init__()
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == 'leaky':
            self.nonlinearity = nn.LeakyReLU()
        elif nonlinearity == 'tanh':
            self.nonlinearity = nn.Tanh()
        else:
            raise('Unknown nonlinearity. Accepting relu or tanh only.')

        self.use_layer_norm = use_layer_norm
        self.dropout = nn.Dropout(dropout_pct)
        self.upsample = nn.Upsample(mode='bilinear', scale_factor=2, align_corners=True)
        self.latent_dim = latent_dim

        # Build Encoder
        encoder_dims = [input_dim] + hidden_dims
        self.encoder = self.build_encoder(encoder_dims, filter_size, padding, stride)

        image_size = 64 # fixed
        flat_dim, self.dec_first_W = self.get_flat_dim(hidden_dims, image_size, 
                                            filter_size, padding, stride)
            
            
        encoder_dims = [flat_dim]+fc_dim
        self.fc_enc = self.build_modules(encoder_dims, self.nonlinearity, dropout_pct)
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(fc_dim[-1])
        
        
        
        self.fc_mu = nn.Linear(fc_dim[-1], latent_dim)
        self.fc_logvar = nn.Linear(fc_dim[-1], latent_dim)
        
        
        fc_dim.reverse()
        decoder_dims = [latent_dim] + fc_dim
        
        self.fc_dec1 = self.build_modules(decoder_dims, self.nonlinearity, dropout_pct)
        self.fc_dec2 = nn.Linear(fc_dim[-1], flat_dim)
        self.dec_first_C = hidden_dims[-1]
        hidden_dims.reverse()
        decoder_dims = hidden_dims
        self.decoder = self.build_decoder(decoder_dims, filter_size, padding, stride)

        # use sigmoid in last layer
        self.final_layer =  nn.Sequential(self.upsample, 
                                          torch.nn.Conv2d(hidden_dims[-1], 
                                                          out_channels = output_dim,
                                                          kernel_size = filter_size, 
                                                          stride = 1, 
                                                          padding = 1),                                          
                                            # nn.BatchNorm2d(h_dim),
                                            nn.Sigmoid(), # NO ACTIVATION
                                            # dropout
                                            )

    @staticmethod
    def get_flat_dim(hidden_dims, W, F, P, S):
        for i in range(len(hidden_dims)):
            W = np.floor((W + 2*P - F)/S + 1)
        return int(hidden_dims[-1] * (W**2)), int(W)


    def build_modules(self,hidden_dims, nonlinearity, dropout_pct):
        modules = []
        input_dim = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim),
                    # nn.BatchNorm1d(h_dim),
                    nonlinearity,
                    nn.Dropout(dropout_pct)
                    )
            )
            input_dim = h_dim
        return nn.Sequential(*modules)
    
    
    def build_encoder(self, hidden_dims, filter_size, padding, stride):
        modules = []
        in_channels = hidden_dims[0]
        for h_dim in hidden_dims[1:]:
            modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, 
                              out_channels = h_dim,
                              kernel_size = filter_size, 
                              stride = stride, 
                              padding = padding
                              # padding_mode = 'replicate'
                              ),
                    # nn.BatchNorm2d(h_dim),
                    self.nonlinearity,
                    # dropout
                    ))
            in_channels = h_dim

        return nn.Sequential(*modules)


    def build_decoder(self, hidden_dims, filter_size, padding, stride):
        nonlinearity=self.nonlinearity
        modules = []
        in_channels = hidden_dims[0]
        for i, h_dim in enumerate(hidden_dims[1:]):
            modules.append(nn.Sequential(
                    self.upsample, 
                    torch.nn.Conv2d(in_channels, 
                              out_channels = h_dim,
                              kernel_size = filter_size, 
                              stride = 1, 
                              padding = 1),
                    # nn.BatchNorm2d(h_dim),
                    nonlinearity,
                    # dropout
                    ))
            in_channels = h_dim

        return nn.Sequential(*modules)

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        result = self.fc_enc(result)
        if self.use_layer_norm:
            result  = self.layer_norm(result)

        mu = self.fc_mu(result)
        log_var = self.fc_logvar(result)
        return [mu, log_var]


    def decode(self, z: Tensor) -> Tensor:
        result = self.fc_dec1(z)
        result = self.fc_dec2(result)

        result = result.reshape((-1, self.dec_first_C, 
                                self.dec_first_W, self.dec_first_W))
        result = self.decoder(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, inpt: Tensor, **kwargs) -> List[Tensor]:
        mu, logvar = self.encode(inpt)
        z = self.reparameterize(mu, logvar)
        out = self.decode(z)
        out = self.final_layer(out)
        return [out, mu, logvar]

    def sample(self, num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim)
            z = z.to(current_device)
            samples = self.decode(z)
            samples = self.final_layer(samples)

        return samples