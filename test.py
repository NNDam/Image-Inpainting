import torch
from models.model import CoarseModel, RefineModel, HyperGraphModel, Discriminator

if __name__ == '__main__':
    model_gen = HyperGraphModel(input_size = 256, coarse_downsample = 3, refine_downsample = 4, channels = 64)
    model_disc = Discriminator(input_size = 256, discriminator_downsample = 6, channels = 64)
    # print(model)
    x = torch.zeros((5, 3, 256, 256))
    y = torch.zeros((5, 1, 256, 256))
    out_coarse, out_refine = model_gen(x, y)
    print(out_coarse.shape, out_refine.shape)

    out_disc = model_disc(x, y)
    print(model_disc)
    print(out_disc.shape)