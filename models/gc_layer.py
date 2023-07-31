import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConvolution(torch.nn.Module) :
    def __init__ (
        self, 
        in_channels,
        out_channels,
        kernel_size,  
        stride=1,  
        padding='same', 
        dilation=1,
        activation='ELU',
        bias = True,
        batch_norm=False,
        negative_slope = 0.2,
    ):
        super ().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        if activation is not None:
            if activation == 'LeakyReLU':
                self.activation = torch.nn.LeakyReLU(negative_slope=negative_slope, inplace=False)
            elif activation == 'ReLU':
                self.activation = torch.nn.ReLU()
            elif activation == 'ELU':
                self.activation = torch.nn.ELU(alpha=1.0, inplace=False)
            else:
                raise NotImplementedError("Could not get activation {}".format(activation))
        else:
            self.activation = None
  
        self.padding = int((kernel_size - 1)/stride) if stride != 1 else padding
        self.conv2d = torch.nn.Conv2d(in_channels = in_channels,
                                        out_channels = out_channels,
                                        kernel_size = kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        padding=self.padding,
                                        bias = bias,
                                        )
        self.mask_conv2d = torch.nn.Conv2d(in_channels = in_channels,
                                    out_channels = out_channels,
                                    kernel_size = kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    padding=self.padding,
                                    bias = bias,
                                    )
        if batch_norm:
            self.batch_norm = torch.nn.BatchNorm2d(out_channels)
        else:
            self.batch_norm = None
        self.sigmoid = torch.nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def __call__ (self, input) :
        # Apply convolution to the Input features
        x = self.conv2d(input) # B C H W
        # If we have final layer then we don't apply any activation
        if self.out_channels == 3 and self.activation is None :
            return x
        mask = self.mask_conv2d(input)
        # Else use gated & activation
        if self.activation is not None:
            x = self.activation(x) * self.gated(mask)
        else:
            #x = x * self.gated(mask)
            raise ValueError("No activation and is not output convolution")

        
        if self.batch_norm is not None:
            return self.batch_norm(x)
        else:
            return x

# Gated Deconvolution layer -> Upsampling + Gated Convolution
class GatedDeConvolution (torch.nn.Module) :
    def __init__ (
            self, 
            in_channels,
            out_channels,
            kernel_size,  
            stride=1,  
            dilation=1,
            padding='same', 
            activation='ELU',
            batch_norm=True):
        super ().__init__()
        self.gate_conv2d = GatedConvolution(in_channels = in_channels,
                                            out_channels = out_channels,
                                            kernel_size = kernel_size,  
                                            stride= stride,  
                                            dilation= dilation,
                                            padding=padding, 
                                            activation=activation,
                                            batch_norm=batch_norm)
        self.upsample = torch.nn.Upsample(scale_factor=2)

    def __call__ (self, input) :
        x = self.upsample(input)
        x = self.gate_conv2d(x)
        return x