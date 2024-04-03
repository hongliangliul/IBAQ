# -*- coding: utf-8 -*-

import json
import os
import shutil
from time import time
import cv2
import config
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from classifier_models import ResNet18,PreActResNet18
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import PostTensorTransform, get_dataloader, CSVDataset
from utils.utils import progress_bar
from torchvision import datasets, transforms, models

from PIL import Image
import copy
import random
import cupy as cp


def get_model(opt):
    netC = None
    optimizerC = None
    schedulerC = None
    if opt.dataset == "ISIC2019":
        netC = PreActResNet18(num_classes=opt.num_classes).to(opt.device)
    else:
        print('erro dataset')    # Optimizer
    optimizerC = torch.optim.SGD(netC.parameters(), opt.lr_C, momentum=0.9, weight_decay=5e-4)
    #optimizerC = torch.optim.Adam(netC.parameters(), lr=opt.lr_C, weight_decay=5e-4)
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

    ratio = 0.1
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

def create_bd(inputs, opt):
    bs, _, _, _ = inputs.shape

    transforms_list = []
    transforms_list.append(transforms.Resize((opt.input_height, opt.input_width)))
    transforms_list.append(transforms.ToTensor())
    transforms_class = transforms.Compose(transforms_list)

    im_target = Image.open(opt.target_img).convert('RGB')
    im_target = transforms_class(im_target)

    im_target = np.clip(im_target.numpy() * 255, 0, 255)
    im_target = torch.from_numpy(im_target).repeat(bs, 1, 1, 1)

    inputs = np.clip(inputs.numpy() * 255, 0, 255)

    bd_inputs = Fourier_pattern(inputs, im_target, opt.beta, opt.alpha)
    bd_inputs = torch.tensor(np.clip(bd_inputs / 255, 0, 1), dtype=torch.float32)

    return bd_inputs.to(opt.device)


def create_cross(inputs, opt):
    cross_inputs = inputs
    return cross_inputs.to(opt.device)

def train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt):
    print(" Train:")
    netC.train()
    rate_bd = opt.pc
    total_loss_ce = 0
    total_sample = 0

    total_clean = 0
    total_bd = 0
    total_cross = 0
    total_clean_correct = 0
    total_bd_correct = 0
    total_cross_correct = 0
    criterion_CE = torch.nn.CrossEntropyLoss()

    transforms = PostTensorTransform(opt).to(opt.device)
    total_time = 0

    avg_acc_cross = 0

    for batch_idx, (inputs, targets) in enumerate(train_dl):
        optimizerC.zero_grad()

        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        bs = inputs.shape[0]

        # Create poisoned data
        num_bd = int(bs * rate_bd)
        num_cross = int(num_bd * opt.cross_ratio)

        if opt.attack_mode == "all2one":
            targets_bd = torch.ones_like(targets[:num_bd]) * opt.target_label
        if opt.attack_mode == "all2all":
            targets_bd = torch.remainder(targets[:num_bd], opt.num_classes)

        inputs_bd = create_bd(copy.deepcopy(inputs[:num_bd].cpu()), opt)
        inputs_cross = create_cross(copy.deepcopy(inputs[num_bd:(num_bd + num_cross)].cpu()), opt)
        # combination
        total_inputs = torch.cat([inputs_bd, inputs_cross, inputs[(num_bd + num_cross):]], dim=0)
        total_inputs = transforms(total_inputs)
        total_targets = torch.cat([targets_bd, targets[num_bd:]], dim=0)

        # training
        total_preds = netC(total_inputs)

        loss_ce = criterion_CE(total_preds, total_targets)

        loss = loss_ce
        loss.backward()

        optimizerC.step()

        total_sample += bs
        total_loss_ce += loss_ce.detach()

        total_clean += bs - num_bd - num_cross
        total_bd += num_bd
        total_cross += num_cross
        total_clean_correct += torch.sum(
            torch.argmax(total_preds[(num_bd + num_cross):], dim=1) == total_targets[(num_bd + num_cross):]
        )
        total_bd_correct += torch.sum(torch.argmax(total_preds[:num_bd], dim=1) == targets_bd)
        if num_cross:
            total_cross_correct += torch.sum(
                torch.argmax(total_preds[num_bd: (num_bd + num_cross)], dim=1)
                == total_targets[num_bd: (num_bd + num_cross)]
            )
            avg_acc_cross = total_cross_correct * 100.0 / total_cross

        avg_acc_clean = total_clean_correct * 100.0 / total_clean
        avg_acc_bd = total_bd_correct * 100.0 / total_bd

        avg_loss_ce = total_loss_ce / total_sample

        if num_cross:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f}|Clean Acc: {:.4f}|Bd Acc: {:.4f}|Cross Acc: {:.4f}".format(
                    avg_loss_ce, avg_acc_clean, avg_acc_bd, avg_acc_cross
                ),
            )
        else:
            progress_bar(
                batch_idx,
                len(train_dl),
                "CE Loss: {:.4f} | Clean Acc: {:.4f} | Bd Acc: {:.4f} ".format(avg_loss_ce, avg_acc_clean, avg_acc_bd),
            )

        # Save image for debugging
        if not batch_idx % 1:
            if not os.path.exists(opt.temps):
                os.makedirs(opt.temps)
            path = os.path.join(opt.temps, "traing.png")
            torchvision.utils.save_image(inputs_bd, path, normalize=True)

        # Images for tensorboard
        if batch_idx == len(train_dl) - 2:
            residual = inputs_bd - inputs[:num_bd]
            batch_img = torch.cat([inputs[:num_bd], inputs_bd, total_inputs[:num_bd], residual], dim=2)

            batch_img = F.upsample(batch_img, scale_factor=(4, 4))
            grid = torchvision.utils.make_grid(batch_img, normalize=True)

    # for tensorboard
    if not epoch % 1:
        tf_writer.add_scalars(
            "Clean Accuracy", {"Clean": avg_acc_clean, "Bd": avg_acc_bd, "Cross": avg_acc_cross}, epoch
        )
        tf_writer.add_image("Images", grid, global_step=epoch)

    schedulerC.step()


def eval(
        netC,
        optimizerC,
        schedulerC,
        test_dl,
        best_clean_acc,
        best_bd_acc,
        best_cross_acc,
        tf_writer,
        epoch,
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
                total_cross_correct += torch.sum(torch.argmax(preds_cross, 1) == targets)

                acc_cross = total_cross_correct * 100.0 / total_sample

                info_string = (
                    "Clean Acc: {:.4f} - B{:.4f} |Bd Acc: {:.4f} - Best: {:.4f}| Cross: {:.4f}".format(
                        acc_clean, best_clean_acc, acc_bd, best_bd_acc, acc_cross, best_cross_acc
                    )
                )
            else:
                info_string = "Clean Acc: {:.4f} - Best: {:.4f}           |           Bd Acc: {:.4f} - Best: {:.4f}".format(
                    acc_clean, best_clean_acc, acc_bd, best_bd_acc
                )
            progress_bar(batch_idx, len(test_dl), info_string)

    # tensorboard
    if not epoch % 1:
        tf_writer.add_scalars("Test Accuracy", {"Clean": acc_clean, "Bd": acc_bd}, epoch)

    if acc_clean > best_clean_acc - 1.0 and acc_bd > best_bd_acc and acc_cross > best_cross_acc - 1.0:
        print(" Saving for best_acc_bd:{:.3f}".format(acc_bd))
        best_bd_acc = acc_bd
        state_dict = {
            "netC": netC.state_dict(),
            "schedulerC": schedulerC.state_dict(),
            "optimizerC": optimizerC.state_dict(),
            "best_clean_acc": best_clean_acc,
            "best_bd_acc": best_bd_acc,
            "best_cross_acc": best_cross_acc,
            "epoch_current": epoch,

        }
        torch.save(state_dict, os.path.join(opt.ckpt_folder, 'best_acc_bd_ckpt.pth.tar'))

    return best_clean_acc, best_bd_acc, best_cross_acc


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
    if opt.dataset == 'ISIC2019':
        train_dl = get_dataloader(opt, True, set_ISIC2019='Train', pretensor_transform=False)
        test_dl = get_dataloader(opt, False, set_ISIC2019='Val', pretensor_transform=False)
    else:
        train_dl = get_dataloader(opt, True)  # True是确认训练还是测试  ，pre_transform默认为false不使用
        test_dl = get_dataloader(opt, False)

    # prepare model
    netC, optimizerC, schedulerC = get_model(opt)

    # Load pretrained model
    mode = opt.attack_mode
    opt.ckpt_folder = os.path.join(opt.checkpoints, opt.dataset, mode + str(opt.experiment_idx))
    opt.log_dir = os.path.join(opt.ckpt_folder, "log_dir")
    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    opt.ckpt_path = os.path.join(opt.ckpt_folder, "best_acc_bd_ckpt.pth.tar")
    if opt.continue_training:
        if os.path.exists(opt.ckpt_path):
            print("Continue training!!")
            state_dict = torch.load(opt.ckpt_path)
            netC.load_state_dict(state_dict["netC"])
            optimizerC.load_state_dict(state_dict["optimizerC"])
            schedulerC.load_state_dict(state_dict["schedulerC"])
            best_clean_acc = state_dict["best_clean_acc"]
            best_bd_acc = state_dict["best_bd_acc"]
            best_cross_acc = state_dict["best_cross_acc"]
            epoch_current = state_dict["epoch_current"]

            tf_writer = SummaryWriter(log_dir=opt.log_dir)
        else:
            print("Pretrained model doesnt exist")
            exit()
    else:
        print("Train from scratch!!!")
        best_clean_acc = 0.0
        best_bd_acc = 0.0
        best_cross_acc = 0.0
        epoch_current = 0

        shutil.rmtree(opt.ckpt_folder, ignore_errors=True)
        os.makedirs(opt.log_dir)
        with open(os.path.join(opt.ckpt_folder, "opt.json"), "w+") as f:
            json.dump(opt.__dict__, f, indent=2)
        tf_writer = SummaryWriter(log_dir=opt.log_dir)

    for epoch in range(epoch_current, opt.n_iters):
        print("Epoch {}:".format(epoch + 1))
        train(netC, optimizerC, schedulerC, train_dl, tf_writer, epoch, opt)
        best_clean_acc, best_bd_acc, best_cross_acc = eval(
            netC,
            optimizerC,
            schedulerC,
            test_dl,

            best_clean_acc,
            best_bd_acc,
            best_cross_acc,
            tf_writer,
            epoch,
            opt,
        )


if __name__ == "__main__":
    main()

