import  os
import cv2
import numpy as np
from PIL import Image
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import datasets, models, transforms
from io import BytesIO
import random

def irregular_mask (image_height, image_width, batch_size=1, min_strokes=16, max_strokes=48) :
    masks = []
    
    for b in range (batch_size) :
        mask = np.zeros ((image_height, image_width), np.uint8)
        mask_shape = mask.shape

        max_width = 20
        number = random.randint (min_strokes, max_strokes)
        for _ in range (number) :
            model = random.random()
            if model < 0.6:
                # Draw random lines
                x1, x2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[0])
                y1, y2 = random.randint(1, mask_shape[1]), random.randint(1, mask_shape[1])
                thickness = random.randint(4, max_width)
                cv2.line(mask, (x1, y1), (x2, y2), (1, 1, 1), thickness)

            elif model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                radius = random.randint(4, max_width)
                cv2.circle(mask, (x1, y1), radius, (1, 1, 1), -1)

            elif model > 0.8:
                # Draw random ellipses
                x1, y1 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                s1, s2 = random.randint(1, mask_shape[0]), random.randint(1, mask_shape[1])
                a1, a2, a3 = random.randint(3, 180), random.randint(3, 180), random.randint(3, 180)
                thickness = random.randint(4, max_width)
                cv2.ellipse(mask, (x1, y1), (s1, s2), a1, a2, a3, (1, 1, 1), thickness)
        
        masks.append (mask[:, :, np.newaxis])

    return masks

def center_mask (image_height, image_width, batch_size=1) :
    mask = np.zeros ((batch_size, image_height, image_width, 1)).astype ('float32')
    mask [:, image_height//4:(image_height//4)*3, image_height//4:(image_height//4)*3, :] = 1.0

    return mask

def mask_border(image_height, image_width, batch_size=1, maximum = 0.5, minimum = 0.3):
    mask = np.zeros ((batch_size, image_height, image_width, 1)).astype ('float32')
    mask_shape = (image_height, image_width)
    for b in range(batch_size):
        to_mask_y = random.randint(int(minimum*mask_shape[0]), int(maximum*mask_shape[0]))

        to_mask_x1 = random.randint(0, int(minimum*mask_shape[1]))
        to_mask_x2 = random.randint(0, int(minimum*mask_shape[1]))
        if to_mask_x2 == 0:
            mask[b, -to_mask_y:, to_mask_x1:, :] = 1
        else:
            mask[b, -to_mask_y:, to_mask_x1:-to_mask_x2, :] = 1
    return mask

def mask_center(image_height, image_width, batch_size=1, min_area = 0.2, max_area = 0.45):
    mask = np.zeros((batch_size, image_height, image_width, 1)).astype ('float32')
    mask_shape = (image_height, image_width)
    for b in range(batch_size):
        area       = int(random.uniform(min_area, max_area)*mask_shape[0]*mask_shape[1])
        mask_width  = random.randint(int(area/mask_shape[1])+16, mask_shape[1]-16)
        mask_height = max(16, int(area/mask_width))
        dx = random.randint(0, mask_shape[1] - mask_width)
        dy = random.randint(0, mask_shape[0] - mask_height)
        mask[b, dy:dy+mask_height, dx:dx+mask_width, :] = 1
    return mask

def save_images (input_image, ground_truth, prediction_coarse, prediction_refine, path) :

    display_list = [input_image, ground_truth, prediction_coarse, prediction_refine]
    img = np.concatenate (display_list, axis=1)
    plt.imsave (path, np.clip (img, 0, 1.0))

def transform_JPEGcompression(image, compress_range = (30, 100)):
    '''
        Perform random JPEG Compression
    '''
    if random.random() < 0.15:
        assert compress_range[0] < compress_range[1], "Lower and higher value not accepted: {} vs {}".format(compress_range[0], compress_range[1])
        jpegcompress_value = random.randint(compress_range[0], compress_range[1])
        out = BytesIO()
        image.save(out, 'JPEG', quality=jpegcompress_value)
        out.seek(0)
        rgb_image = Image.open(out)
        return rgb_image
    else:
        return image

def transform_gaussian_noise(img_pil, mean = 0.0, var = 10.0):
    '''
        Perform random gaussian noise
    '''
    if random.random() < 0.15:
        img = np.array(img_pil)
        height, width, channels = img.shape
        sigma = var**0.5
        gauss = np.random.normal(mean, sigma,(height, width, channels))
        noisy = img + gauss
        cv2.normalize(noisy, noisy, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy = noisy.astype(np.uint8)
        return Image.fromarray(noisy)
    else:
        return img_pil

def transform_resize(image, resize_range = (32, 112), target_size = 112):
    if random.random() < 0.15:
        assert resize_range[0] < resize_range[1], "Lower and higher value not accepted: {} vs {}".format(resize_range[0], resize_range[1])
        resize_value = random.randint(resize_range[0], resize_range[1])
        resize_image = image.resize((resize_value, resize_value))
        return resize_image.resize((target_size, target_size))
    else:
        return image



class FaceInpaintingData(Dataset):
    def __init__(self, gt_folder, input_size = (256, 256)):
        super(FaceInpaintingData, self).__init__()
        self.paths = []
        self.gt_folder = gt_folder
        for path in self.gt_folder.split(','):
            self.paths += glob.glob(path + '/*.jpg') 
            self.paths += glob.glob(path + '/*.png') 
        self.paths = sorted(self.paths)
        self.input_size = input_size
        self.transforms = transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        bgr = cv2.imread(path)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Random crop
        inters =  [cv2.INTER_NEAREST , cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
        crop_x1 = random.randint(0, 20) if random.random() > 0.5 else 0
        crop_y1 = random.randint(0, 20) if random.random() > 0.5 else 0
        crop_x2 = random.randint(0, 20) if random.random() > 0.5 else 0
        crop_y2 = random.randint(0, 20) if random.random() > 0.5 else 0
        try:
            rgb = cv2.resize(rgb[crop_y1: height - crop_y2, crop_x1: width - crop_x2], self.input_size, interpolation= random.choice(inters))
        except:
            rgb = cv2.resize(rgb, self.input_size, interpolation= random.choice(inters))

        flag = random.random()
        if flag > 0.66:
            mask = irregular_mask(self.input_size[1], self.input_size[0], batch_size=1)[0]
        elif flag > 0.33:
            mask = mask_border(self.input_size[1], self.input_size[0], batch_size=1)[0]
        else:
            mask = center_mask(self.input_size[1], self.input_size[0], batch_size=1)[0]

        rgb = np.array(transform_JPEGcompression(Image.fromarray(rgb)))
        # masked = torch.from_numpy((rgb*(1.0 - mask)).astype('float32')/255.0)
        # Build masked image
        masked = np.where(mask==1, 255.0, rgb)
        masked = torch.from_numpy(masked.astype('float32')/255.0)
        rgb = torch.from_numpy(rgb.astype('float32')/255.0)
        # Transform & normalize
        mask = torch.from_numpy(mask.astype('float32')) 
        return masked.permute(2, 0, 1), mask.permute(2, 0, 1), rgb.permute(2, 0, 1)
        # return image, title