'''
Author: myzhibei myzhibei@qq.com
Date: 2023-05-17 21:15:42
LastEditors: myzhibei myzhibei@qq.com
LastEditTime: 2023-05-28 12:42:26
FilePath: \猫狗分类\CatsDogsVs\CatsDogVS_pth.py
Description: 

Copyright (c) 2023 by myzhibei myzhibei@qq.com, All Rights Reserved. 
'''
from multiprocessing import freeze_support
import sys
import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader


# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter

# Image datasets and image manipulation
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms

# Image display
import matplotlib.pyplot as plt
import numpy as np


import platform
plat = platform.system().lower()
if plat == 'windows':
    log_path = r"D:\Log"  # windows tensorbroad不支持中文路径
elif plat == 'linux':
    log_path = r"./logs"  # 日志文件路径

dataset_path = r"./data"  # 数据集存放路径
model_save_path = r"./model"  # 模型待存储路径

# logf = open('./logs/runCNN.log', 'a')
# sys.stdout = logf

print(time.time())


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


writer = SummaryWriter(log_path)
# To view, start TensorBoard on the command line with:
# tensorboard --logdir="D:\Log" --bind_all
# ...and open a browser tab to http://localhost:6006/


classes = [
    "cat",
    "dog",
]


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


def load_data():
    import getdata
    # 下载数据集
    # Download training data from open datasets.
    training_data = getdata.DogsVSCatsDataset('train', dataset_path)
    vld_data = getdata.DogsVSCatsDataset('validation', dataset_path)

    # Download test data from open datasets.
    test_data = getdata.DogsVSCatsDataset('test', dataset_path)

    batch_size = 1
    workers = 1

    # Create data loaders.
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, num_workers=workers, drop_last=True)
    vld_dataloader = DataLoader(
        vld_data, batch_size=batch_size, num_workers=workers, drop_last=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)
    # for X, y in test_dataloader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break

    # Extract a batch of 4 images
    dataiter = iter(train_dataloader)
    images, labels = next(dataiter)

    # Create a grid from the images and show them
    img_grid = torchvision.utils.make_grid(images)
    matplotlib_imshow(img_grid, one_channel=False)
    print("show image")

    # Write image data to TensorBoard log dir
    writer.add_image('CATs OR DOGs Images', img_grid)
    writer.flush()
    return train_dataloader, vld_dataloader, test_dataloader


# Define model
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.CNN_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(7),
        )
        self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(64*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.CNN_stack(x)
        x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    global total_train_step
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_train_step = total_train_step + 1
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(
                f"train_step:{total_train_step} \t loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            writer.add_scalars('CDTraining Loss', {
                               'Training': loss}, total_train_step)


def vld(dataloader, model, loss_fn):
    global total_test_step
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # print(f"num_batches = {num_batches}")
    model.eval()
    test_loss, correct = 0, 0
    total_test_step = total_test_step + 1
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    writer.add_scalars('CDValidation Loss', {
        'Validation': test_loss}, total_test_step)
    writer.add_scalar("Validation accuracy", correct, total_test_step)


# 保存模型
def save_model(CNN_model):
    # model_name = "MNIST_CNN_model" + \
    #     time.strftime("%Y%m%d%H%I%S", time.localtime(time.time()))+".pth"
    model_name = "CatsDogVS_model.pth"
    model_path = model_save_path + '/' + model_name
    torch.save(CNN_model.state_dict(), model_path)
    print(f"Saved PyTorch CNN Model State to {model_path}")


def test_saved_model():
    CNN_model = CNN().to(device)
    CNN_model.load_state_dict(torch.load(model_path))
    CNN_model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.unsqueeze(0)
        x = x.to(device)
        pred = CNN_model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


if __name__ == '__main__':
    freeze_support()
    train_dataloader, vld_dataloader, test_dataloader = load_data()
    starttime = time.time()
    global total_train_step  # 记录训练的次数
    total_train_step = 0
    global total_test_step  # 记录测试的次数
    total_test_step = 0
    epochs = 1
    CNN_model = CNN().to(device)
    print(CNN_model)

    # input = torch.ones((5, 1, 28, 28), dtype=torch.float32).to(device)  # 测试输入 用于初步检测网络最后的输出形状
    # writer.add_graph(CNN_model, input)  # 获得网络结构图
    # writer.flush()

    model_path = "./model/CatsDogVS_model.pth"
    if os.path.exists(model_path):
        print(f"Load model {model_path}")
        CNN_model.load_state_dict(torch.load(model_path))

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MultiLabelSoftMarginLoss()
    optimizer = torch.optim.SGD(CNN_model.parameters(), lr=1e-3, momentum=0.9)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, CNN_model, loss_fn, optimizer)
        # vld(vld_dataloader, CNN_model, loss_fn)
    save_model(CNN_model)

    endtime = time.time()
    print("Done!")
    writer.flush()
    writer.close()
    print(f"Time-consuming: {(endtime - starttime)} \n")
    # save_model()
    # test_saved_model()
    print('Finished')
    logf.close()
