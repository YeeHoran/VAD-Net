'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function


# import torch
# import torch.nn as nn
import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import torchvision
import transforms as transforms
import numpy as np
import os
import argparse
import utils
import utils2
from fer import FER2013
# from CK import CK
# from torch.autograd import Variable
from models import *
# add by HY

from models.resnet_reg2 import ResNet18RegressionTwoOutputs
import pandas as pd
import torch.utils.data
print("run from begin1")
print("run from begin2")


# add by HY

def custom_transform(crops):
    return torch.stack([transforms.ToTensor()(crop) for crop in crops])

print("run from begin3")
# Training
def train(epoch, trainloader):
    print('\nEpoch: %d' % epoch)
    global list_Train_AveLoss
    # global trainset
    # global trainloader
    net.train()
    total_loss = 0.0
    total_samples = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    ###########################
    # for batch_idx, (inputs, targets) in enumerate(trainloader):  ###########问题在这里，每次都要加载数据。？？?，先用小数据集测试。
    for batch in trainloader:
        inputs, targets = batch
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs = outputs.squeeze(dim=1)  # remove the dimension with size 1, resulting in a tensor of size [16].
        ##########
        # to be consistent with 'target' size.
        # add by HY, for orthogonal convolution
        ########################################################################################
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
        ########################
        loss = criterion(outputs, targets)
        print(loss)
        loss = loss + 0.5 * diff
        loss = loss.to(torch.float32)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()

        # Update statistics
        total_loss = total_loss + loss.item()
        total_samples = total_samples + targets.size(0)
        '''utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (total_loss / (batch_idx + 1), 100. * correct / total, correct, total))'''
    average_loss = total_loss / total_samples
    print(f'Average Train Loss: {average_loss:.3f} ')
    list_Train_AveLoss.append(average_loss)


def PublicTest(epoch, PublicTestloader):
    global best_PublicTest_Averageloss
    global best_PublicTest_Averageloss_epoch

    global list_Pubtest_AveLoss

    net.eval()
    PublicTest_loss = 0
    total_Pubsamples = 0

    # for batch_idx, (inputs, targets) in enumerate(PublicTestloader):
    for batch in PublicTestloader:
        inputs, targets = batch
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        ##########
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        ############
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PublicTest_loss += loss.data
        total_Pubsamples += targets.size(0)
        '''
        #utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                           % (PublicTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''
    # Save checkpoint.
    PublicTest_av_loss = PublicTest_loss / total_Pubsamples
    PublicTest_av_loss = PublicTest_av_loss.item()
    list_Pubtest_AveLoss.append(PublicTest_av_loss)
    print(f'PublicTest_av_loss: {PublicTest_av_loss:.3f} ')
    if PublicTest_av_loss < best_PublicTest_Averageloss:
        best_PublicTest_Averageloss = PublicTest_av_loss
        best_PublicTest_Averageloss_epoch = epoch
        print('Saving..')
        print("best_PublicTest_Averageloss: %0.3f" % PublicTest_av_loss)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'loss': PublicTest_av_loss,
            'epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PublicTest_model.t7'))
        best_PublicTest_Averageloss = PublicTest_av_loss
        best_PublicTest_Averageloss_epoch = epoch


def PrivateTest(epoch, PrivateTestloader):
    global best_PublicTest_Averageloss
    global best_PublicTest_Averageloss_epoch
    global best_PrivateTest_Averageloss
    global best_PrivateTest_Averageloss_epoch

    global list_Pritest_AveLoss

    net.eval()
    PrivateTest_loss = 0
    total_PriSamples = 0
    # for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
    for batch in PrivateTestloader:
        inputs, targets = batch
        bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        #############################
        with torch.no_grad():
            inputs, targets = Variable(inputs), Variable(targets)
        ###############################
        outputs = net(inputs)
        outputs_avg = outputs.view(bs, ncrops, -1).mean(1)  # avg over crops
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.data
        total_PriSamples += targets.size(0)
        '''
        utils.progress_bar(batch_idx, len(PublicTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                   % (PrivateTest_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        '''
    PrivateTest_av_loss = PrivateTest_loss / total_PriSamples
    PrivateTest_av_loss = PrivateTest_av_loss.item()
    list_Pritest_AveLoss.append(PrivateTest_av_loss)
    print(f'PrivateTest_av_loss: {PrivateTest_av_loss:.3f} ')
    if PrivateTest_av_loss < best_PrivateTest_Averageloss:
        best_PrivateTest_Averageloss = PrivateTest_av_loss
        best_PrivateTest_Averageloss_epoch = epoch
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_av_loss)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PublicTest_Averageloss': best_PublicTest_Averageloss,
            'best_PrivateTest_Averageloss': best_PrivateTest_Averageloss,
            'best_PublicTest_Averageloss_epoch': best_PublicTest_Averageloss_epoch,
            'best_PrivateTest_Averageloss_epoch': best_PrivateTest_Averageloss_epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path, 'PrivateTest_model.t7'))


if __name__ == '__main__':
    print("run from main1")
    parser = argparse.ArgumentParser(description='PyTorch VAD-Fer2013 Regression Training')
    parser.add_argument('--model', type=str, default='ResNet18RegressionTwoOutputs', help='CNN architecture')
    parser.add_argument('--dataset', type=str, default='FER2013', help='CNN architecture')  ###############
    parser.add_argument('--bs', default=128, type=int, help='learning rate')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--resume', '-r', action='store_true', default='false', help='resume from checkpoint')
    opt = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    best_PublicTest_Averageloss = torch.tensor(float('inf'), dtype=torch.float32)
    best_PublicTest_Averageloss_epoch = 0
    best_PrivateTest_Averageloss = torch.tensor(float('inf'), dtype=torch.float32)  # best PrivateTest accuracy
    best_PrivateTest_Averageloss_epoch = 0
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    list_Train_AveLoss = []
    list_Pubtest_AveLoss = []
    list_Pritest_AveLoss = []

    learning_rate_decay_start = 80  # 50
    learning_rate_decay_every = 5  # 5
    learning_rate_decay_rate = 0.9  # 0.9

    cut_size = 44
    # total_epoch = 250
    total_epoch = 120

    path = os.path.join(opt.dataset + '_' + opt.model)

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(44),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.TenCrop(cut_size),
        custom_transform,
    ])

    trainset = FER2013(split='Training', transform=transform_train)  ################
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.bs, shuffle=True, num_workers=1)
    PublicTestset = FER2013(split='PublicTest', transform=transform_test)  ###############################
    PublicTestloader = torch.utils.data.DataLoader(PublicTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
    PrivateTestset = FER2013(split='PrivateTest', transform=transform_test)  ########################
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.bs, shuffle=False, num_workers=1)
    print("run from main2")
    # print(opt.model)
    # if opt.model == 'VGG19':
    # net = VGG('VGG19')
    # else:
    net = ResNet18RegressionTwoOutputs()
    #################################################################
    '''
    if opt.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir(path), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(path, 'PrivateTest_model.t7'))

        net.load_state_dict(checkpoint['net'])
        best_PublicTest_Averageloss = checkpoint['best_PublicTest_Averageloss']
        best_PrivateTest_Averageloss = checkpoint['best_PrivateTest_Averageloss']
        best_PublicTest_Averageloss_epoch = checkpoint['best_PublicTest_Averageloss_epoch']
        best_PrivateTest_Averageloss_epoch = checkpoint['best_PrivateTest_Averageloss_epoch']
        start_epoch = checkpoint['best_PrivateTest_Averageloss_epoch'] + 1
    else:
    '''
    print('==> Building model..')
    ####################################################################
    if use_cuda:
        net.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

    for epoch in range(start_epoch, total_epoch):
        train(epoch, trainloader)
        PublicTest(epoch, PublicTestloader)
        PrivateTest(epoch, PrivateTestloader)
        # add by HY
        # save the process of loss for each epoch
        data = open("data.txt", 'a')
        print("best_PublicTest_Averageloss: %0.3f" % best_PublicTest_Averageloss, file=data)
        print("best_PublicTest_Averageloss_epoch: %d" % best_PublicTest_Averageloss_epoch, file=data)
        print("best_PrivateTest_Averageloss: %0.3f" % best_PrivateTest_Averageloss, file=data)
        print("best_PrivateTest_Averageloss_epoch: %d" % best_PrivateTest_Averageloss_epoch, file=data)
        data.close()
        # add by HY
    # print out the best model's parameters.
    print("best_PublicTest_Averageloss: %0.3f" % best_PublicTest_Averageloss)
    print("best_PublicTest_Averageloss_epoch: %d" % best_PublicTest_Averageloss_epoch)
    print("best_PrivateTest_Averageloss: %0.3f" % best_PrivateTest_Averageloss)
    print("best_PrivateTest_Averageloss_epoch: %d" % best_PrivateTest_Averageloss_epoch)
    # save best model's parameters to file as well.
    data = open("data.txt", 'a')
    print("best model is:", file=data)
    print("best_PublicTest_Averageloss: %0.3f" % best_PublicTest_Averageloss, file=data)
    print("best_PublicTest_Averageloss_epoch: %d" % best_PublicTest_Averageloss_epoch, file=data)
    print("best_PrivateTest_Averageloss: %0.3f" % best_PrivateTest_Averageloss, file=data)
    print("best_PrivateTest_Averageloss_epoch: %d" % best_PrivateTest_Averageloss_epoch, file=data)
    data.close()
    # save the average loss in each epoch for train, publictest, and privatetest to a csv file to investigate.
    column_heads = ['TrainAveLoss', 'PubtestAveLoss', 'PritestAveLoss']
    # Create a DataFrame
    df = pd.DataFrame(list(zip(list_Train_AveLoss, list_Pubtest_AveLoss, list_Pritest_AveLoss)), columns=column_heads)
    # Specify the file path
    csv_file_path = 'AveLossProcess.csv'
    # Save the DataFrame to a CSV file
    df.to_csv(csv_file_path, index=False)
