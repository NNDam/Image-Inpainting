import os
import time
import torch
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from losses import PerceptualLoss, SobelLoss, GANLoss
from utils import cosine_lr, get_lr
from models.model import CoarseModel, RefineModel, HyperGraphModel, Discriminator, VGGStyleDiscriminator
from dataloader import FaceInpaintingData
from torch.utils.data import Dataset, DataLoader

def discriminator_loss(disc_original_output, disc_generated_output):
    B = disc_original_output.size()[0]
    loss_object = torch.nn.BCEWithLogitsLoss()
    disc_original_output = disc_original_output.view(B, -1)
    disc_generated_output = disc_generated_output.view(B, -1)
    real_loss = loss_object(disc_original_output, torch.ones_like(disc_original_output))
    fake_loss = loss_object(disc_generated_output, torch.zeros_like(disc_generated_output))
    # real_loss = gan_loss_obj(disc_original_output, target_is_real = True, is_disc = True)
    # fake_loss = gan_loss_obj(disc_generated_output, target_is_real = False, is_disc = True)
    # # print(disc_original_output, disc_generated_output, real_loss, fake_loss)
    # real_score = disc_original_output.detach().mean()
    # fake_score = disc_generated_output.detach().mean()
    return real_loss, fake_loss, real_loss + fake_loss

def generator_loss(disc_generated_output, gen_output_coarse, gen_output_refine, target, mask):
    # gen_output_coarse = gen_output_refine = B x 3 x H x W
    loss_object = torch.nn.BCEWithLogitsLoss()
    B = disc_generated_output.size()[0]
    disc_generated_output = disc_generated_output.view(B, -1)
    gan_loss = GAN_LOSS_WEIGHT * loss_object(disc_generated_output, torch.ones_like(disc_generated_output))
    # gan_loss =  gan_loss_obj(disc_generated_output, target_is_real = True, is_disc = False)

    

    # # Permute
    # gen_output_coarse = gen_output_coarse.permute(0, 2, 3, 1)
    # gen_output_refine = gen_output_refine.permute(0, 2, 3, 1)
    # target = target.permute(0, 2, 3, 1)
    # mask   = mask.permute(0, 2, 3, 1)

    # print(gen_output_coarse.shape, gen_output_refine.shape, target.shape, mask.shape)

    # Hole
    hole_l1_loss =  torch.mean(torch.abs((mask) * (target - gen_output_coarse))) * 0.5
    hole_l1_loss += torch.mean(torch.abs((mask) * (target - gen_output_refine)))
    hole_l1_loss *= HOLE_LOSS_WEIGHT

    # Valid
    valid_l1_loss =  torch.mean(torch.abs((1 - mask) * (target - gen_output_coarse))) * 0.5
    valid_l1_loss += torch.mean(torch.abs((1 - mask) * (target - gen_output_refine)))
    valid_l1_loss *= VALID_LOSS_WEIGHT

    

    # perceptual_loss_refine, _     = percep_loss_obj(gen_output_refine, target)
    # perceptual_loss_out    = PERCEPTUAL_LOSS_OUT_WEIGHT *  perceptual_loss_refine
    # Perceptual
    perceptual_loss_coarse, _     = percep_loss_obj(gen_output_coarse, target)
    perceptual_loss_refine, _     = percep_loss_obj(gen_output_refine, target)
    perceptual_loss_out    = PERCEPTUAL_LOSS_OUT_WEIGHT * (0.5*perceptual_loss_coarse + perceptual_loss_refine)/1.5
    
    img_comp   = (gen_output_refine * mask + target * (1-mask)) # in range [0.0, 1.0]
    perceptual_loss_comp, _    =   percep_loss_obj(img_comp, target)
    perceptual_loss_comp *= PERCEPTUAL_LOSS_COMP_WEIGHT

    # Edge loss
    edge_coarse  = sobel_loss_obj(gen_output_coarse, target) * 0.5
    edge_out     = sobel_loss_obj(gen_output_refine, target)
    edge_loss = EDGE_LOSS_WEIGHT * (edge_out + edge_coarse)
    # edge_loss = 0

    total_loss =  valid_l1_loss \
                    + hole_l1_loss \
                    + gan_loss \
                    + perceptual_loss_out \
                    + perceptual_loss_comp \
                    + edge_loss
    return total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, perceptual_loss_out, perceptual_loss_comp

def eval(step):
    print('-'*50)
    print('Evaluating step {} ...'.format(step))
    model_gen.eval()
    counter = 0
    vis_folder = os.path.join(training_dir, "visualize", str(step))
    os.makedirs(vis_folder, exist_ok = True)
    for i, batch in enumerate(val_loader):
        inputs, masks, targets = batch
        inputs  = inputs.to('cuda')
        masks   = masks.to('cuda')
        targets = targets.to('cuda')

        with torch.no_grad():
            prediction_coarse, prediction_refine = model_gen(inputs, masks)
            prediction_coarse = prediction_coarse.detach().cpu().numpy()
            prediction_refine = prediction_refine.detach().cpu().numpy()
            inputs = inputs.detach().cpu().numpy()
            masks  = masks.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()
        for ix in range(len(inputs)):
            img_input = np.array(inputs[ix]*255.0, dtype = np.uint8)
            img_mask = np.tile(np.array(masks[ix]*255.0, dtype = np.uint8), (3, 1, 1))
            img_coarse = np.array(prediction_coarse[ix]*255.0, dtype = np.uint8)
            img_refine = np.array(prediction_refine[ix]*255.0, dtype = np.uint8)
            img_target = np.array(targets[ix]*255.0, dtype = np.uint8)
            img_mask_bin = img_mask/255.0
            img_complete = np.array(img_target*(1 - img_mask_bin) + img_refine*img_mask_bin, dtype = np.uint8)
            img = np.concatenate([img_input, img_mask, img_coarse, img_refine, img_complete, img_target], axis = 2)
            img = np.transpose(img, (1, 2, 0))
            cv2.imwrite(os.path.join(vis_folder, "vis_{}.jpg".format(counter + ix)), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        counter += len(inputs)
    model_gen.train()

def train():
    step = 0
    for epoch in range (0, epoches) :
       
        os.makedirs(os.path.join (training_dir, f"EPOCH{epoch}"), exist_ok = True)

        avg_total_loss = 0
        avg_valid_l1_loss = 0
        avg_hole_l1_loss = 0
        avg_edge_loss = 0
        avg_gan_loss = 0
        avg_pl_out = 0
        avg_pl_comp = 0
        avg_disc_loss = 0
        avg_real_loss = 0
        avg_fake_loss = 0
        print ("EPOCH : " + str (epoch))
        
        for i, batch in enumerate(train_loader):
            if len(batch[0]) != batch_size:
                print('[WARNING] Skipped batch {} due to invalid number/batch_size:'.format(i), len(batch[0]), batch_size)
                continue

            step += 1
            lr = scheduler_gen(step)
            _ = scheduler_disc(step)

            # torch.autograd.set_detect_anomaly(True)
            i0 = time.time()
            # Optimize generator
            for p in model_disc.parameters():
                p.requires_grad = False
            optimizer_gen.zero_grad()
            
            inputs, masks, targets = batch
            inputs  = inputs.to('cuda')
            masks   = masks.to('cuda')
            targets = targets.to('cuda')
            prediction_coarse, prediction_refine = model_gen(inputs, masks)
            # prediction_coarse = torch.clamp(prediction_coarse, 0.0, 1.0)
            # prediction_refine = torch.clamp(prediction_refine, 0.0, 1.0)
            disc_generated_output = model_disc(prediction_refine, masks)
            total_loss, valid_l1_loss, hole_l1_loss, edge_loss, gan_loss, pl_out, pl_comp = generator_loss(disc_generated_output, prediction_coarse, prediction_refine, targets, masks)
            total_loss.backward()
            optimizer_gen.step()
            i1 = time.time()
            # Optimize discriminator
            for p in model_disc.parameters():
                p.requires_grad = True
            optimizer_disc.zero_grad()
            fake_d_pred  = model_disc(prediction_refine.detach(), masks) # .detach() here mean disable gradient from generator
            real_d_pred  = model_disc(targets, masks)

            real_score, fake_score, disc_loss = discriminator_loss(real_d_pred, fake_d_pred)
            disc_loss.backward()
            # torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer_disc.step()
            i2 = time.time()
            avg_total_loss += total_loss.detach().cpu().numpy()
            avg_valid_l1_loss += valid_l1_loss.detach().cpu().numpy()
            avg_hole_l1_loss += hole_l1_loss.detach().cpu().numpy()
            avg_edge_loss += edge_loss.detach().cpu().numpy()
            avg_gan_loss += gan_loss.detach().cpu().numpy()
            avg_pl_out += pl_out.detach().cpu().numpy()
            avg_pl_comp += pl_comp.detach().cpu().numpy()
            avg_disc_loss += disc_loss.detach().cpu().numpy()
            avg_real_loss += real_score.detach().cpu().numpy()
            avg_fake_loss += fake_score.detach().cpu().numpy()
            i3 = time.time()
            if step % print_every == 0 and step > 1:
                print('-'*50)
                tavg_total_loss = avg_total_loss/(i + 1)
                tavg_valid_l1_loss = avg_valid_l1_loss/(i + 1)
                tavg_hole_l1_loss = avg_hole_l1_loss/(i + 1)
                tavg_edge_loss = avg_edge_loss/(i + 1)
                tavg_gan_loss = avg_gan_loss/(i + 1)
                tavg_pl_out = avg_pl_out/(i + 1)
                tavg_pl_comp = avg_pl_comp/(i + 1)
                tavg_disc_loss = avg_disc_loss/(i + 1)
                tavg_real_loss = avg_real_loss/(i + 1)
                tavg_fake_loss = avg_fake_loss/(i + 1)
                print ('step', step)
                print ('avg_gen_loss = ', tavg_total_loss)
                print ('avg_valid_l1_loss = ', tavg_valid_l1_loss)
                print ('avg_hole_l1_loss = ', tavg_hole_l1_loss)
                print ('avg_edge_loss = ', tavg_edge_loss)
                print ('avg_gan_loss = ', tavg_gan_loss)
                print ('avg_pl_out = ', tavg_pl_out)
                print ('avg_pl_comp = ', tavg_pl_comp)
                print ('avg_disc_loss = ', tavg_disc_loss)
                print ('avg_real_loss = ', tavg_real_loss)
                print ('avg_fake_loss = ', tavg_fake_loss)
                print ('lr_gen = ', get_lr(optimizer_gen))
                print ('lr_disc = ', get_lr(optimizer_disc))
                print ('time data / time gen / time disc / time all = {} / {} / {} / {}'.format(i0 - i4, i1-i0, i2-i1,i3-i0))
            if step % valid_every == 0 and step > 0:
                print('VALIDATE')
                eval(step)
            i4 = time.time()

        avg_total_loss /= (i + 1)
        avg_valid_l1_loss /= (i + 1)
        avg_hole_l1_loss /= (i + 1)
        avg_edge_loss /= (i + 1)
        avg_gan_loss /= (i + 1)
        avg_pl_out /= (i + 1)
        avg_pl_comp /= (i + 1)
        avg_disc_loss /= (i + 1)
        avg_real_loss /= (i + 1)
        avg_fake_loss /= (i + 1)
        print ('avg_total_loss = ', avg_total_loss)
        print ('avg_valid_l1_loss = ', avg_valid_l1_loss)
        print ('avg_hole_l1_loss = ', avg_hole_l1_loss)
        print ('avg_edge_loss = ', avg_edge_loss)
        print ('avg_gan_loss = ', avg_gan_loss)
        print ('avg_pl_out = ', avg_pl_out)
        print ('avg_pl_comp = ', avg_pl_comp)
        print ('avg_disc_loss = ', avg_disc_loss)
        print ('avg_real_loss = ', avg_real_loss)
        print ('avg_fake_loss = ', avg_fake_loss)

if __name__ == '__main__':
    # Config
    valid_every = 500
    print_every = 10
    batch_size = 2
    lr_gen = 1e-4
    lr_disc = 1e-4
    wd = 0.01
    warmup_length = 0 # 50k iter 
    epoches = 1000
    num_workers = 2
    train_gt_folder = ''
    val_gt_folder = '/home/damnguyen/dataset/StyleGAN40k_valid_256'
    training_dir = 'experiments'

    VALID_LOSS_WEIGHT = 0.2
    HOLE_LOSS_WEIGHT = 1.0
    EDGE_LOSS_WEIGHT = 0.05
    GAN_LOSS_WEIGHT = 0.002
    PERCEPTUAL_LOSS_OUT_WEIGHT = 0.0001
    PERCEPTUAL_LOSS_COMP_WEIGHT = 0.0001
    # PERCEPTUAL_LOSS_COARSE_WEIGHT = 0.0
    # PERCEPTUAL_LOSS_OUT_WEIGHT = 0.0
    # PERCEPTUAL_LOSS_COMP_WEIGHT = 0.0

    # Define model 
    model_gen = HyperGraphModel(input_size = 256, coarse_downsample = 3, refine_downsample = 4, channels = 64)
    model_disc = Discriminator(input_size = 256, discriminator_downsample = 6, channels = 64)
    # model_disc = VGGStyleDiscriminator(num_in_ch = 3, num_feat = 16)
    model_gen.to('cuda')
    model_disc.to('cuda')
    # Loss
    percep_loss_obj = PerceptualLoss(layer_weights = {'conv1_2': 0.1,
                                                  'conv2_2': 0.1,
                                                  'conv3_4': 1,
                                                  'conv4_4': 1,
                                                  'conv5_4': 1,
                                                  },
                                vgg_type='vgg19',
                                use_input_norm=True,
                                perceptual_weight=0.1).to('cuda')
    sobel_loss_obj = SobelLoss(loss_weight=1.0, reduction='mean')
    gan_loss_obj = GANLoss('wgan_softplus')
    # Dataloader
    trainset  = FaceInpaintingData(val_gt_folder)
    valset    = FaceInpaintingData(val_gt_folder)
    train_loader = DataLoader(
            trainset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, drop_last=True
        )
    val_loader   = DataLoader(
            valset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, drop_last=True
        )
    num_batches = len(train_loader)

    # Optimizer & Scheduler

    params_gen = [p for name, p in model_gen.named_parameters()]
    params_disc = [p for name, p in model_disc.named_parameters()]
    optimizer_gen = torch.optim.AdamW(params_gen, lr=lr_gen, weight_decay=wd)
    optimizer_disc = torch.optim.AdamW(params_disc, lr=lr_disc, weight_decay=wd)

    scheduler_gen = cosine_lr(optimizer_gen, lr_gen, warmup_length, epoches * num_batches)
    scheduler_disc = cosine_lr(optimizer_disc, lr_disc, warmup_length, epoches * num_batches)

    train()