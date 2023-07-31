import torch
import torch.nn as nn
import torch.nn.functional as F

from models.gc_layer import GatedConvolution, GatedDeConvolution
from models.hypergraph_layer import HypergraphConv


class GatedBlock(torch.nn.Module):
    def __init__(self,
                    in_channels = 64,
                    out_channels = 128,
                    n_conv = 2,
                    downscale_first = True,
                    dilation = 1,
                    activation = 'LeakyReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv = n_conv

        # First conv
        first_stride = 2 if downscale_first else 1
        self.first_conv = GatedConvolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=first_stride,
                                            dilation=1,
                                            padding='same',
                                            activation=activation)
        self.rest_conv = nn.ModuleList()
        for i in range(n_conv):
            self.rest_conv.append(
                GatedConvolution(in_channels=out_channels,
                                out_channels=out_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=dilation,
                                padding='same',
                                activation=activation)
            )

    def forward(self, x):
        x = self.first_conv(x)
        for i in range(self.n_conv):
            x = self.rest_conv[i](x)
        return x

class GatedDeBlock(torch.nn.Module):
    def __init__(self,
                    in_channels = 64,
                    out_channels = 32,
                    n_conv = 2,
                    activation = 'LeakyReLU',
                    dilation = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_conv = n_conv

        self.first_conv = nn.ModuleList()
        for i in range(n_conv):
            self.first_conv.append(
                GatedConvolution(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=dilation,
                                padding='same',
                                activation=activation)
            )

        self.last_conv = GatedDeConvolution(in_channels=in_channels,
                                            out_channels=out_channels,
                                            kernel_size=3,
                                            stride=1,
                                            dilation=1,
                                            padding='same',
                                            activation=activation)
        

    def forward(self, x):
        x = self.first_conv[0](x)
        for i in range(1, self.n_conv):
            x = self.first_conv[i](x)
        x = self.last_conv(x)
        return x

class CoarseModel(torch.nn.Module):
    def __init__(self, input_size = 256, channels = 64, downsample = 3):
        super().__init__()
        self.downsample = downsample

        self.conv1 = GatedConvolution (in_channels=4, out_channels=channels, kernel_size=7, stride=1, dilation=1, padding='same', activation='LeakyReLU') # RGB + Mask

        # Encoder For Coarse Network
        self.enc_convs = nn.ModuleList()
        in_channels = channels
        for i in range(self.downsample):
            self.enc_convs.append(GatedBlock(
                            in_channels = in_channels,
                            out_channels = 2*in_channels,
                            n_conv = 2,
                            downscale_first = True,
                            dilation = 1
                        ))
            in_channels = 2*in_channels

        # Center Convolutions for higher receptive field
        # These convolutions are with dilation=2
        self.mid_convs = nn.ModuleList()
        for i in range(3):
            self.mid_convs.append(
                    GatedConvolution(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=1,
                                dilation=2,
                                padding='same',
                                activation='LeakyReLU')
                )

        # Decoder Network for Coarse Network
        self.dec_convs = nn.ModuleList()
        for i in range (self.downsample):
            if i > 0:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = 2*in_channels,  # Skip connection from Encoder
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))
            else:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = in_channels,
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))

            in_channels = int(in_channels//2)

        # assert in_channels == channels, "Invalid configuration {} vs {}".format(in_channels, channels)

        self.last_dec   = GatedConvolution(in_channels=in_channels,
                                out_channels=channels,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation='LeakyReLU')

        self.coarse_out = GatedConvolution(in_channels=channels,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation=None)

    def forward(self, x):
        x = self.conv1(x)
        skip_layer = []
        for i in range(self.downsample):
            x = self.enc_convs[i](x)
            if i != self.downsample - 1:
                skip_layer.append(x)
        for i in range(3):
            x = self.mid_convs[i](x)
        for i in range(self.downsample):
            if i > 0:
                skip_layer_idx = self.downsample - 1 - i
                x = torch.cat([x, skip_layer[skip_layer_idx]], dim = 1)
            x = self.dec_convs[i](x)
        x = self.last_dec(x)
        x = self.coarse_out(x)
        return x




class RefineModel(torch.nn.Module):
    def __init__(self, input_size = 256, channels = 64, downsample = 4):
        super().__init__()
        self.input_size = input_size
        self.downsample = downsample

        self.conv1 = GatedConvolution(in_channels=4, out_channels=channels, kernel_size=7, stride=1, dilation=1, padding='same', activation='LeakyReLU') # RGB + Mask

        # Encoder For Coarse Network
        self.enc_convs = nn.ModuleList()
        in_channels = channels
        for i in range(self.downsample):
            if i != self.downsample - 1:
                self.enc_convs.append(GatedBlock(
                                in_channels = in_channels,
                                out_channels = 2*in_channels,
                                n_conv = 2,
                                downscale_first = True
                            ))
            else:
                # Last encoder layer use 2 dilation and 3 block
                # padding: o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
                #          16 = 16 + 2*p - 3 - 2 + 1
                self.enc_convs.append(GatedBlock(
                                in_channels = in_channels,
                                out_channels = 2*in_channels,
                                n_conv = 3,
                                downscale_first = True,
                                dilation = 2,
                            ))
            in_channels = 2*in_channels

        # Apply Hypergraph convolution on skip connections
        self.hypergraph_convs = nn.ModuleList()
        for i in range(1, self.downsample):
            hyp_channels = (2**i)*channels
            hyp_size = int(self.input_size/(2**i))
            if i == 1:
                # First downscale use GateConvolution as skip-connection
                self.hypergraph_convs.append(
                        GatedConvolution(in_channels=2*channels,
                                        out_channels=2*channels,
                                        kernel_size=3,
                                        stride=1,
                                        dilation=1,
                                        padding='same',
                                        activation='LeakyReLU') 
                    )
            else:
                # All downscale bellow use Hypergraph as skip-connection
                self.hypergraph_convs.append(
                        HypergraphConv (in_channels=hyp_channels,
                                        out_channels=2*hyp_channels,
                                        features_height=hyp_size,
                                        features_width=hyp_size,
                                        edges=256,
                                        filters=128,
                                        apply_bias=True,
                                        trainable=True,
                                        activation = 'LeakyReLU')
                    )

        # Decoder Network for Refine Network
        self.dec_convs = nn.ModuleList()
        for i in range (self.downsample):
            if i == 0:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = in_channels,
                                out_channels = int(in_channels//2),
                                n_conv = 3,
                                dilation = 2,
                            ))
            elif i == self.downsample - 1:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = 2*in_channels,  # Skip connection from Encoder
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))
            else:
                self.dec_convs.append(GatedDeBlock(
                                in_channels = 3*in_channels,  # Skip connection from Encoder
                                out_channels = int(in_channels//2),
                                n_conv = 2,
                            ))
            
            

            in_channels = int(in_channels//2)

        # assert in_channels == channels, "Invalid configuration {} vs {}".format(in_channels, channels)

        self.last_dec   = GatedConvolution(in_channels=in_channels,
                                out_channels=channels,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation='LeakyReLU')

        self.refine_out = GatedConvolution(in_channels=channels,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                dilation=1,
                                padding='same',
                                activation=None)

    def forward(self, x):
        x = self.conv1(x)
        skip_layer = []
        for i in range(self.downsample):
            x = self.enc_convs[i](x)
            if i != self.downsample - 1: # Ignore last downscale
                hyp = self.hypergraph_convs[i](x)
                skip_layer.append(hyp)
 
        for i in range(self.downsample):
            # Ignore skip-connection on first decoder layer
            if i > 0:
                skip_layer_idx = self.downsample - 1 - i
                x = torch.cat([x, skip_layer[skip_layer_idx]], dim = 1)

            x = self.dec_convs[i](x)
        x = self.last_dec(x)
        x = self.refine_out(x)
        return x

class HyperGraphModel(torch.nn.Module):
    def __init__(self, input_size = 256, coarse_downsample = 3, refine_downsample = 4, channels = 64):
        super().__init__()
        self.coarse_model = CoarseModel(input_size = input_size,
                                        downsample = coarse_downsample,
                                        channels = channels)
        self.refine_model = RefineModel(input_size = input_size,
                                        downsample = refine_downsample,
                                        channels = channels)

    # Generator Network
    def forward(self, img, mask):
        # mask: 0 - original image, 1.0 - masked
        inp_coarse = torch.cat([img, mask], dim = 1)
        out_coarse = self.coarse_model(inp_coarse)
        out_coarse = torch.clamp(out_coarse, min = 0.0, max = 1.0)
        b, _, h, w = mask.size()
        mask_rp = mask.repeat(1, 3, 1, 1)
        inp_refine = out_coarse * mask_rp + img * (1.0 - mask_rp)
        inp_refine = torch.cat([inp_refine, mask], dim = 1)
        out_refine = self.refine_model(inp_refine)
        out_refine = torch.clamp(out_refine, min = 0.0, max = 1.0)
        return out_coarse, out_refine

class Discriminator(torch.nn.Module):
    def __init__(self, input_size = 256, discriminator_downsample = 6, channels = 64):
        super().__init__()
        self.input_size = input_size
        self.discriminator_downsample = discriminator_downsample
        self.channels = channels

        # First convolution
        self.conv1 = GatedConvolution(in_channels=4, out_channels=channels, kernel_size=5, stride=2, dilation=1, padding='same', activation='LeakyReLU')
        # Other convolution
        self.enc_convs = nn.ModuleList()
        in_channels = self.channels
        for i in range(1, self.discriminator_downsample):
            mult = (2**i) if (2**i) < 8 else 8
            out_channels = self.channels * mult
            self.enc_convs.append(GatedConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, dilation=1, padding='same', activation='LeakyReLU'))
            in_channels = out_channels
        # Last convolution
        # self.last_conv = GatedConvolution(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, dilation=1, padding='same', activation=None)
        # self.fc1   = torch.nn.Linear(out_channels* 4 * 4, 128)
        # self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # self.fc2   = torch.nn.Linear(128, 1)

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim = 1)
        x = self.conv1(x)
        for i in range(self.discriminator_downsample-1):
            x = self.enc_convs[i](x)
        # x = x.view(x.size(0), -1)
        # x = self.lrelu(self.fc1(x))
        # x = self.fc2(x)
        return x

class VGGStyleDiscriminator(nn.Module):
    """VGG style discriminator with input size 256 x 256.

    It is now used to train VideoGAN.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features.
            Default: 64.
    """

    def __init__(self, num_in_ch, num_feat):
        super(VGGStyleDiscriminator, self).__init__()

        self.conv0_0 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(num_feat, num_feat, 4, 2, 1, bias=False)
        self.bn0_1 = nn.SyncBatchNorm(num_feat, affine=True)

        self.conv1_0 = nn.Conv2d(num_feat, num_feat * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.SyncBatchNorm(num_feat * 2, affine=True)
        self.conv1_1 = nn.Conv2d(num_feat * 2, num_feat * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.SyncBatchNorm(num_feat * 2, affine=True)

        self.conv2_0 = nn.Conv2d(num_feat * 2, num_feat * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.SyncBatchNorm(num_feat * 4, affine=True)
        self.conv2_1 = nn.Conv2d(num_feat * 4, num_feat * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.SyncBatchNorm(num_feat * 4, affine=True)

        self.conv3_0 = nn.Conv2d(num_feat * 4, num_feat * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv3_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.conv4_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv4_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.conv5_0 = nn.Conv2d(num_feat * 8, num_feat * 8, 3, 1, 1, bias=False)
        self.bn5_0 = nn.SyncBatchNorm(num_feat * 8, affine=True)
        self.conv5_1 = nn.Conv2d(num_feat * 8, num_feat * 8, 4, 2, 1, bias=False)
        self.bn5_1 = nn.SyncBatchNorm(num_feat * 8, affine=True)

        self.linear1 = nn.Linear(num_feat * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 256 and x.size(3) == 256, (f'Input spatial size must be 256x256, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: (128, 128)

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: (64, 64)

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: (32, 32)

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: (16, 16)

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: (8, 8)

        feat = self.lrelu(self.bn5_0(self.conv5_0(feat)))
        feat = self.lrelu(self.bn5_1(self.conv5_1(feat)))  # output spatial size: (4, 4)

        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out