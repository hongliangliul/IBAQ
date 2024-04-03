import json
import os
import shutil
from time import time

import config
import numpy as np
import torch
import torch.nn.functional as F
from classifier_models import PreActResNet18
from torch import nn
import cv2

from utils.dataloader import PostTensorTransform, get_dataloader
from utils.utils import progress_bar
from torchvision import datasets, transforms, models

import copy
from PIL import Image
import cupy as cp

def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None

   
    if opt.dataset == "ISIC2019":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)

    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)

    # Scheduler
    schedulerC = torch.optim.lr_scheduler.MultiStepLR(optimizerC, opt.schedulerC_milestones, opt.schedulerC_lambda)

    return netC, optimizerC, schedulerC


def enheng(image,trigger):
    fc = image
    fc2 = trigger
    # 获取幅度和相位信息
    f = np.fft.fft2(fc)
    f2 = np.fft.fft2(fc2)

    # 交换源图像的幅度部分和目标图像的幅度谱
    amp_source_shift = np.fft.fftshift(f)
    amp_target_shift = np.fft.fftshift(f2)
    # amp_source_shift = amp_source
    # amp_target_shift = amp_target
    amp_source, pha_source = np.abs(amp_source_shift), np.angle(amp_source_shift)
    amp_target, pha_target = np.abs(amp_target_shift), np.angle(amp_target_shift)
    h, w = image.shape
    b = (np.floor(np.amin((h, w)) * 0.1)).astype(int)
    c_h = np.floor(h / 2.0).astype(int)
    c_w = np.floor(w / 2.0).astype(int)

    h1 = c_h - b
    h2 = c_h + b + 1
    w1 = c_w - b
    w2 = c_w + b + 1

    ratio = 0.15
    replaced_phase = pha_target[h1:h2, w1:w2]
    amp_source_shift[h1:h2, w1:w2] = ratio * amp_target[h1:h2, w1:w2] * np.exp(1j * replaced_phase) * np.exp(
        1j * replaced_phase) + (1 - ratio) * amp_source_shift[h1:h2, w1:w2]
    # 反向变换得到处理后的频域图像
    amp_source_shift = np.fft.ifftshift(amp_source_shift)
    local_in_trg = np.fft.ifft2(amp_source_shift)

    local_in_trg = np.real(local_in_trg)
    return local_in_trg
def Fourier_pattern(img_, target_img, beta, ratio):
    #  get the amplitude and phase spectrum of trigger image
    img_ = np.asarray(img_)
    target_img = np.asarray(target_img)
    img_ = np.transpose(img_, (0, 2, 3, 1))
    target_img=np.transpose( target_img, (0, 2, 3, 1))
    num_images, H, W, C = img_.shape
    processed_images = np.zeros((num_images, H, W, C))
    for i in range(num_images):
        image = img_[i]
        #img_int = np.floor(image).astype(np.uint8)
        trigger = target_img[i]
        #trigger_int = np.floor(trigger).astype(np.uint8)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        tycc = cv2.cvtColor(trigger, cv2.COLOR_RGB2YCrCb)
        ty = tycc[:, :, 0]
        y = ycrcb[:, :, 0]
        #trigger_int = cv2.resize(trigger_int, (224, 224))
        encoded_image = enheng(y,ty)
        ycrcb[:, :, 0] = encoded_image
        output_rgb_image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        output_rgb_image = np.clip(output_rgb_image, 0, 255)
        processed_images[i] = output_rgb_image
    processed_images = np.transpose(processed_images, (0, 3, 1, 2))
    return np.asarray(processed_images)
def create_bd(inputs,  opt):

    bs,_ ,_ ,_ = inputs.shape

    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())  
    transforms_class = transforms.Compose(transforms_list)

    im_target = Image.open(opt.target_img).convert('RGB')
    im_target = transforms_class(im_target)

    im_target = np.clip(im_target.numpy() * 255, 0, 255)
    im_target = torch.from_numpy(im_target).repeat(bs,1,1,1)

    inputs = np.clip(inputs.numpy()*255,0,255)

    bd_inputs = Fourier_pattern(inputs,im_target,opt.beta,opt.alpha)

    bd_inputs = torch.tensor(np.clip(bd_inputs/255,0,1),dtype=torch.float32)


    return bd_inputs.to(opt.device)

def create_cross(inputs, opt):
    bs, _, _, _ = inputs.shape
    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())  
    transforms_class = transforms.Compose(transforms_list)

    ims_noise = []
    noiseImage_list = os.listdir(opt.cross_dir)
    noiseImage_names = np.random.choice(noiseImage_list,bs)
    for noiseImage_name in noiseImage_names:
        noiseImage_path = os.path.join(opt.cross_dir,noiseImage_name)

        im_noise = Image.open(noiseImage_path).convert('RGB')
        im_noise = transforms_class(im_noise)
        im_noise = np.clip(im_noise.numpy()*255,0,255)
        ims_noise.append(im_noise)

    inputs = np.clip(inputs.numpy()*255,0,255)
    ims_noise = np.array(ims_noise)
    cross_inputs = Fourier_pattern(inputs, ims_noise, opt.beta, opt.alpha)
    cross_inputs = torch.tensor(np.clip(cross_inputs/255,0,1),dtype=torch.float32)

    return cross_inputs.to(opt.device)

def eval(
    netC,
    test_dl,
    opt,
):
    print(" Eval:")
    netC.eval()

    total_sample = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0

    for batch_idx, (inputs, targets) in enumerate(test_dl):
        with torch.no_grad():
            inputs, targets = inputs.to(opt.device), targets.to(opt.device)
            bs = inputs.shape[0]
            total_sample += bs

            # Evaluate Clean data
            preds_clean = netC(inputs)
            total_clean_correct += torch.sum(torch.argmax(preds_clean, 1) == targets)

            # Evaluate Backdoor data

            inputs_bd = create_bd(copy.deepcopy(inputs.cpu()), opt)
            if opt.attack_mode == "all2one":
                targets_bd = torch.ones_like(targets) * opt.target_label
            if opt.attack_mode == "all2all":
                targets_bd = torch.remainder(targets, opt.num_classes)
            preds_bd = netC(inputs_bd)
            total_bd_correct += torch.sum(torch.argmax(preds_bd, 1) == targets_bd)

            acc_clean = total_clean_correct * 100.0 / total_sample
            acc_bd = total_bd_correct * 100.0 / total_sample

            # Evaluate cross
            if opt.cross_ratio:

                inputs_cross = create_cross(copy.deepcopy(inputs.cpu()), opt)
                preds_cross = netC(inputs_cross)
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets_bd)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = "BA: {:.4f}      |       ASR: {:.4f}       |         P-ASR: {:.4f}".format(acc_clean, acc_bd, acc_cross)
            else:
                info_string = "BA: {:.4f} - Best: {:.4f}        |        ASR: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)


def main():
    # parameter prepare
    opt = config.get_arguments().parse_args()

    if opt.dataset == 'ISIC2019':
        opt.num_classes = 10
    else:
        raise Exception("Invalid Dataset")

    if opt.dataset == "ISIC2019":
        opt.input_height = 32
        opt.input_width = 32
        opt.input_channel = 3
    else:
        raise Exception("Invalid Dataset")

    # Dataset
    test_dl = get_dataloader(opt,False, set_ISIC2019='Test', pretensor_transform=False)
    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    
    opt.ckpt_path = opt.test_model

    if os.path.exists(opt.ckpt_path):
        state_dict = torch.load(opt.ckpt_path)
        netC.load_state_dict(state_dict["netC"])
    else:
        print("Pretrained model doesnt exist")
        exit()

    

    eval(
        netC,
        test_dl,
        opt,
    )


if __name__ == "__main__":
    main()
