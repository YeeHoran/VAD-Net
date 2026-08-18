'''Train Fer2013 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
import utils2
from fer import FER2013
from torch.autograd import Variable
from models import *
from models.resnet_reg2 import ResNet18RegressionTwoOutputs
import pandas as pd
import torch.utils.data
import glob
import sys
import pickle
import warnings

# 忽略TypedStorage警告
warnings.filterwarnings("ignore", category=UserWarning, module="transforms.functional")


def custom_transform(crops):
    """将多个crop堆叠成张量"""
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])


# Training
def train(epoch, trainloader):
    global list_Train_AveMSE
    net.train()
    total_loss = 0.0
    total_samples = 0

    # 学习率衰减
    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)
    else:
        current_lr = opt.lr

    for batch in trainloader:
        inputs, targets = batch

        # 确保targets是1维张量 [batch_size]
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        # 前向传播
        outputs = net(inputs)

        # 确保输出是1维 [batch_size]
        if outputs.dim() == 2 and outputs.size(1) == 1:
            outputs = outputs.squeeze(1)

        # 正交正则化
        diff = utils2.orth_dist(net.layer2[0].shortcut[0].weight) + utils2.orth_dist(
            net.layer3[0].shortcut[0].weight) + utils2.orth_dist(net.layer4[0].shortcut[0].weight)
        diff += utils2.deconv_orth_dist(net.layer1[0].conv1.weight, stride=1) + utils2.deconv_orth_dist(
            net.layer1[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net.layer2[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net.layer2[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net.layer3[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net.layer3[1].conv1.weight, stride=1)
        diff += utils2.deconv_orth_dist(net.layer4[0].conv1.weight, stride=2) + utils2.deconv_orth_dist(
            net.layer4[1].conv1.weight, stride=1)

        # 计算损失
        loss = criterion(outputs, targets)
        loss = loss + 0.5 * diff
        loss = loss.to(torch.float32)

        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        # 更新统计
        total_loss += loss.item()
        total_samples += targets.size(0)

    # 计算平均MSE和RMSE
    average_mse = total_loss / total_samples
    average_rmse = np.sqrt(average_mse)
    print(f'Epoch {epoch}: Train MSE={average_mse:.6f}, RMSE={average_rmse:.6f}, lr={current_lr:.6f}')
    list_Train_AveMSE.append(average_mse)


def PublicTest(epoch, PublicTestloader):
    global best_PublicTest_AverageRMSE
    global best_PublicTest_AverageRMSE_epoch
    global list_Pubtest_AveRMSE

    net.eval()
    PublicTest_loss = 0.0
    total_Pubsamples = 0

    for batch in PublicTestloader:
        inputs, targets = batch

        # 处理10-crop
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        # 确保targets是1维
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = net(inputs)
            # 对10个crop取平均
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            # 确保输出是1维
            if outputs_avg.dim() == 2 and outputs_avg.size(1) == 1:
                outputs_avg = outputs_avg.squeeze(1)

            loss = criterion(outputs_avg, targets)
            PublicTest_loss += loss.item()
            total_Pubsamples += targets.size(0)

    # 计算平均MSE和RMSE
    PublicTest_av_mse = PublicTest_loss / total_Pubsamples
    PublicTest_av_rmse = np.sqrt(PublicTest_av_mse)

    list_Pubtest_AveRMSE.append(PublicTest_av_rmse)
    print(f'Epoch {epoch}: PublicTest MSE={PublicTest_av_mse:.6f}, RMSE={PublicTest_av_rmse:.6f}')

    # 基于RMSE选择最佳模型
    if PublicTest_av_rmse < best_PublicTest_AverageRMSE:
        best_PublicTest_AverageRMSE = PublicTest_av_rmse
        best_PublicTest_AverageRMSE_epoch = epoch
        print(f'  -> New best PublicTest RMSE: {PublicTest_av_rmse:.6f}')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
            'best_public_mse': PublicTest_av_mse,
            'best_public_rmse': PublicTest_av_rmse,
            'epoch': epoch,
            'train_mse_list': list_Train_AveMSE,
            'pubtest_rmse_list': list_Pubtest_AveRMSE,
            'pritest_rmse_list': list_Pritest_AveRMSE,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_AverageRMSE = PublicTest_av_rmse
        best_PublicTest_AverageRMSE_epoch = epoch


def PrivateTest(epoch, PrivateTestloader):
    global best_PublicTest_AverageRMSE
    global best_PublicTest_AverageRMSE_epoch
    global best_PrivateTest_AverageRMSE
    global best_PrivateTest_AverageRMSE_epoch
    global list_Pritest_AveRMSE

    net.eval()
    PrivateTest_loss = 0.0
    total_PriSamples = 0

    for batch in PrivateTestloader:
        inputs, targets = batch

        # 处理10-crop
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        # 确保targets是1维
        if targets.dim() == 2 and targets.size(1) == 1:
            targets = targets.squeeze(1)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        with torch.no_grad():
            outputs = net(inputs)
            outputs_avg = outputs.view(bs, ncrops, -1).mean(1)
            # 确保输出是1维
            if outputs_avg.dim() == 2 and outputs_avg.size(1) == 1:
                outputs_avg = outputs_avg.squeeze(1)

            loss = criterion(outputs_avg, targets)
            PrivateTest_loss += loss.item()
            total_PriSamples += targets.size(0)

    # 计算平均MSE和RMSE
    PrivateTest_av_mse = PrivateTest_loss / total_PriSamples
    PrivateTest_av_rmse = np.sqrt(PrivateTest_av_mse)

    list_Pritest_AveRMSE.append(PrivateTest_av_rmse)
    print(f'Epoch {epoch}: PrivateTest MSE={PrivateTest_av_mse:.6f}, RMSE={PrivateTest_av_rmse:.6f}')

    # 基于RMSE选择最佳模型
    if PrivateTest_av_rmse < best_PrivateTest_AverageRMSE:
        best_PrivateTest_AverageRMSE = PrivateTest_av_rmse
        best_PrivateTest_AverageRMSE_epoch = epoch
        print(f'  -> New best PrivateTest RMSE: {PrivateTest_av_rmse:.6f}')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
            'best_public_rmse': best_PublicTest_AverageRMSE,
            'best_private_rmse': best_PrivateTest_AverageRMSE,
            'best_public_rmse_epoch': best_PublicTest_AverageRMSE_epoch,
            'best_private_rmse_epoch': best_PrivateTest_AverageRMSE_epoch,
            'epoch': epoch,
            'train_mse_list': list_Train_AveMSE,
            'pubtest_rmse_list': list_Pubtest_AveRMSE,
            'pritest_rmse_list': list_Pritest_AveRMSE,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))

        # 保存最新checkpoint
        latest_state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
            'best_public_rmse': best_PublicTest_AverageRMSE,
            'best_private_rmse': best_PrivateTest_AverageRMSE,
            'best_public_rmse_epoch': best_PublicTest_AverageRMSE_epoch,
            'best_private_rmse_epoch': best_PrivateTest_AverageRMSE_epoch,
            'epoch': epoch,
            'train_mse_list': list_Train_AveMSE,
            'pubtest_rmse_list': list_Pubtest_AveRMSE,
            'pritest_rmse_list': list_Pritest_AveRMSE,
        }
        torch.save(latest_state, os.path.join(path, 'latest_checkpoint.t7'))


def check_data_stats(dataloader, name="Dataset", is_test=False):
    """检查数据集的统计信息"""
    all_targets = []
    for batch in dataloader:
        inputs, targets = batch
        all_targets.extend(targets.cpu().numpy().flatten())

    all_targets = np.array(all_targets)
    print(f"{name}: samples={len(all_targets)}, mean={all_targets.mean():.4f}, std={all_targets.std():.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch VAD-Fer2013 Regression Training')
    parser.add_argument('--model', type=str, default='ResNet18RegressionTwoOutputs', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume from checkpoint')
    parser.add_argument('--force_restart', action='store_true', default=True, help='force restart from epoch 0')
    opt = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    # 初始化最佳RMSE
    best_PublicTest_AverageRMSE = torch.tensor(float('inf'), dtype=torch.float32)
    best_PublicTest_AverageRMSE_epoch = 0
    best_PrivateTest_AverageRMSE = torch.tensor(float('inf'), dtype=torch.float32)
    best_PrivateTest_AverageRMSE_epoch = 0
    start_epoch = 0

    # 存储损失值
    list_Train_AveMSE = []
    list_Pubtest_AveRMSE = []
    list_Pritest_AveRMSE = []

    # 超参数设置
    learning_rate_decay_start = 80
    learning_rate_decay_every = 5
    learning_rate_decay_rate = 0.9
    cut_size = 44
    total_epoch = 120

    path = os.path.join(opt.dataset + '_' + opt.model)

    if not os.path.isdir(path):
        os.makedirs(path)

    # ========== 数据准备 ==========
    print('Preparing data...')

    transform_train = transforms.Compose([
        transforms.RandomCrop(cut_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        custom_transform,
    ])

    trainset = FER2013(split='Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=opt.bs, shuffle=True, num_workers=0
    )

    PublicTestset = FER2013(split='PublicTest', transform=transform_test)
    PublicTestloader = torch.utils.data.DataLoader(
        PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=0
    )

    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(
        PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=0
    )

    check_data_stats(trainloader, "Training", is_test=False)
    check_data_stats(PublicTestloader, "PublicTest", is_test=True)
    check_data_stats(PrivateTestloader, "PrivateTest", is_test=True)

    # ========== 模型构建 ==========
    net = ResNet18RegressionTwoOutputs()

    if use_cuda:
        net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    if use_cuda and torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    # 强制从epoch 0开始
    print('Training from scratch (epoch 0)...')
    start_epoch = 0

    if opt.force_restart:
        old_models = glob.glob(os.path.join(path, '*.t7'))
        if old_models:
            for f in old_models:
                try:
                    os.remove(f)
                except:
                    pass

    # ========== 开始训练 ==========
    print(f'Training {total_epoch} epochs...')
    print('=' * 60)

    try:
        for epoch in range(start_epoch, total_epoch):
            train(epoch, trainloader)
            PublicTest(epoch, PublicTestloader)
            PrivateTest(epoch, PrivateTestloader)
            print('-' * 60)

            # 保存训练日志
            with open("data.txt", 'a') as data:
                print(f"Epoch: {epoch}", file=data)
                print(f"best_PublicTest_RMSE: {best_PublicTest_AverageRMSE:.6f}", file=data)
                print(f"best_PublicTest_epoch: {best_PublicTest_AverageRMSE_epoch}", file=data)
                print(f"best_PrivateTest_RMSE: {best_PrivateTest_AverageRMSE:.6f}", file=data)
                print(f"best_PrivateTest_epoch: {best_PrivateTest_AverageRMSE_epoch}", file=data)
                print("-" * 40, file=data)

            # 每5个epoch保存检查点
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(path, f'checkpoint_epoch_{epoch}.t7')
                state = {
                    'net': net.state_dict() if use_cuda else net,
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'train_mse_list': list_Train_AveMSE,
                    'pubtest_rmse_list': list_Pubtest_AveRMSE,
                    'pritest_rmse_list': list_Pritest_AveRMSE,
                }
                torch.save(state, checkpoint_path)

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        checkpoint_path = os.path.join(path, 'interrupted_checkpoint.t7')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_mse_list': list_Train_AveMSE,
            'pubtest_rmse_list': list_Pubtest_AveRMSE,
            'pritest_rmse_list': list_Pritest_AveRMSE,
        }
        torch.save(state, checkpoint_path)
        sys.exit(0)
    except Exception as e:
        print(f"\nTraining error: {e}")
        checkpoint_path = os.path.join(path, 'error_checkpoint.t7')
        state = {
            'net': net.state_dict() if use_cuda else net,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_mse_list': list_Train_AveMSE,
            'pubtest_rmse_list': list_Pubtest_AveRMSE,
            'pritest_rmse_list': list_Pritest_AveRMSE,
        }
        torch.save(state, checkpoint_path)
        raise

    # ========== 输出最终结果 ==========
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best PublicTest RMSE: {best_PublicTest_AverageRMSE:.6f} (Epoch {best_PublicTest_AverageRMSE_epoch})")
    print(f"Best PrivateTest RMSE: {best_PrivateTest_AverageRMSE:.6f} (Epoch {best_PrivateTest_AverageRMSE_epoch})")
    print("=" * 60)

    # 保存结果到文件
    with open("data.txt", 'a') as data:
        print("\n" + "=" * 40, file=data)
        print("FINAL RESULTS:", file=data)
        print(f"Best PublicTest RMSE: {best_PublicTest_AverageRMSE:.6f} (Epoch {best_PublicTest_AverageRMSE_epoch})",
              file=data)
        print(f"Best PrivateTest RMSE: {best_PrivateTest_AverageRMSE:.6f} (Epoch {best_PrivateTest_AverageRMSE_epoch})",
              file=data)

    # 保存损失历史到CSV
    column_heads = ['TrainAveMSE', 'PubtestAveRMSE', 'PritestAveRMSE']
    df = pd.DataFrame(list(zip(list_Train_AveMSE, list_Pubtest_AveRMSE, list_Pritest_AveRMSE)), columns=column_heads)
    csv_file_path = 'AveLossProcess.csv'
    df.to_csv(csv_file_path, index=False)

    print(f"Results saved to: {csv_file_path}, data.txt, {path}/")