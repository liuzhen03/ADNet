#-*- coding:utf-8 _*-  
""" 
@author: LiuZhen
@license: Apache Licence 
@file: utils.py 
@time: 2021/01/25
@contact: liuzhen.pwd@gmail.com
@site:  
@software: PyCharm 

"""
import numpy as np
import cv2
import imageio
from math import log10
import random
import torch
import torch.nn as nn
import torch.nn.init as init
#from skimage.measure.simple_metrics import compare_psnr
imageio.plugins.freeimage.download()


def ReadImages(fileNames):
    imgs = []
    for imgStr in fileNames:
        img=cv2.cvtColor(cv2.imread(imgStr, cv2.IMREAD_UNCHANGED),
                     cv2.COLOR_BGR2RGB) / 255.0
        imgs.append(img)
    return np.array(imgs)


def imread_uint16_png(image_path, alignratio_path):
    # Load the align_ratio variable and ensure is in np.float32 precision
    align_ratio = np.load(alignratio_path).astype(np.float32)
    # Load image without changing bit depth and normalize by align ratio
    return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB) / align_ratio


def imwrite_uint16_png(image_path, image, alignratio_path):
    align_ratio = (2 ** 16 - 1) / image.max()
    np.save(alignratio_path, align_ratio)
    uint16_image_gt = np.round(image * align_ratio).astype(np.uint16)
    cv2.imwrite(image_path, cv2.cvtColor(uint16_image_gt, cv2.COLOR_RGB2BGR))
    return None


def ev_alignment(img, expo, gamma):
    return ((img ** gamma) * 2.0**(-1*expo))**(1/gamma)


def psnr(x, target):
    sqrdErr = np.mean((x - target) ** 2)
    return 10 * log10(1/sqrdErr)


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // args.lr_decay_interval))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_parameters(net):
    """Init layer parameters"""
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


def set_random_seed(seed):
    """Set random seed for reproduce"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def range_compressor(x):
    return (np.log(1 + 5000 * x)) / np.log(1 + 5000)


def range_compressor_tensor(x):
    const_1 = torch.from_numpy(np.array(1.0)).cuda()
    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """
    calculate SSIM

    :param img1: [0, 255]
    :param img2: [0, 255]
    :return:
    """
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def radiance_writer(out_path, image):

    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[...,0], image[...,1]), image[...,2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[...,0:3] = np.around(image[...,0:3] * scaled_mantissa[...,None])
        rgbe[...,3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)


def save_hdr(path, image):
    return radiance_writer(path, image)


def tta(img0_0, img1_0, img2_0, img0_1, img1_1, img2_1):
    data={}
    data[0]= [img0_0, img1_0, img2_0, img0_1, img1_1, img2_1]
    data[1]=[img0_0.permute(0,1,3,2), img1_0.permute(0,1,3,2),
                img2_0.permute(0,1,3,2), img0_1.permute(0,1,3,2), img1_1.permute(0,1,3,2), img2_1.permute(0,1,3,2)]
    data[2]=[img0_0.flip(-1), img1_0.flip(-1), img2_0.flip(-1), img0_1.flip(-1), img1_1.flip(-1), img2_1.flip(-1)]
    data[3]=[img0_0.flip(-2), img1_0.flip(-2), img2_0.flip(-2), img0_1.flip(-2), img1_1.flip(-2), img2_1.flip(-2)]

    return data


def reverse_tta(data):
    tmp=(data[0]+data[1].permute(0,1,3,2)+data[2].flip(-1)+data[3].flip(-2))/4

    return tmp


def mu_tonemap(hdr_image, mu=5000):
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)


def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    return mu_tonemap(hdr_image/norm_value, mu)


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return  mu_tonemap(bounded_hdr, mu)


def psnr_tanh_norm_mu_tonemap(hdr_nonlinear_ref, hdr_nonlinear_res, percentile=99, gamma=2.24):
    hdr_linear_ref = hdr_nonlinear_ref**gamma
    hdr_linear_res = hdr_nonlinear_res**gamma
    norm_perc = np.percentile(hdr_linear_ref, percentile)
    return psnr(tanh_norm_mu_tonemap(hdr_linear_ref, norm_perc), tanh_norm_mu_tonemap(hdr_linear_res, norm_perc))


def psnr(im0, im1):
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))


def normalized_psnr(im0, im1, norm):
    return psnr(im0/norm, im1/norm)
