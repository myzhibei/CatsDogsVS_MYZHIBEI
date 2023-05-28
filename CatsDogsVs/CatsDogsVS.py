
import argparse
import sys
import time
import platform
import os
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
from torch.serialization import save
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter, writer
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=False)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class AverageMeter(object):
    def __init__(self, name, fmt=':.6f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} val: {val' + self.fmt + '} avg: {avg' + \
            self.fmt + '} sum: {sum' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


# Define model
class Net(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        logits = self.classifier(x)
        return logits


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


def train(train_loader, model, loss_fn, optimizer, epoch, writer):
    model.train()
    train_loss_record = AverageMeter(name="train_loss_record")
    train_acc_record = AverageMeter(name="train_acc_record")

    with tqdm(train_loader, desc=f"TRAIN EPOCH: {epoch}") as train_bar:
        for batch_no,(data, target) in enumerate(train_bar):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            acc1 = accuracy(output, target, topk=(1,))

            train_acc_record.update(acc1[0].item(), data.size(0))
            train_loss_record.update(loss.item(), data.size(0))
            writer.add_scalar("train_loss_per_batch", train_loss_record.avg, batch_no + epoch*len(train_loader))
            writer.add_scalar("train_acc_per_batch", train_acc_record.avg, batch_no + epoch*len(train_loader))
            # if batch_no % 10 == 0:
            #     writer.add_scalar("train_loss_per_100batch", train_loss_record.avg, batch_no + epoch*len(train_loader))
            #     writer.add_scalar("train_acc_per_100batch", train_acc_record.avg, batch_no + epoch*len(train_loader))

    writer.add_scalar("train_loss", train_loss_record.avg, epoch)
    writer.add_scalar("train_acc", train_acc_record.avg, epoch)


def validate(val_loader, model, loss_fn, epoch, writer, save_dict):
    model.eval()
    val_loss_record = AverageMeter(name="val_loss_record")
    val_acc_record = AverageMeter(name="val_acc_record")
    with torch.no_grad():
        with tqdm(val_loader, desc=f"VALID EPOCH: {epoch}") as val_bar:
            for (data, target) in val_bar:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = loss_fn(output, target)

                acc1 = accuracy(output, target, topk=(1,))
                val_acc_record.update(acc1[0].item(), data.size(0))
                val_loss_record.update(loss.item(), data.size(0))

    writer.add_scalar("val_loss", val_loss_record.avg, epoch)
    writer.add_scalar("val_acc", val_acc_record.avg, epoch)

    if val_acc_record.avg > save_dict["max"]:
        save_dict.update({"max": val_acc_record.avg,
                         "epoch": epoch, "state_dict": model.state_dict()})


def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


classes = {1: "dog", 0: "cat"}


# 定义预测过程
def test_model(model_save_name, test_dataset_path):

    model = Net(num_classes=2)
    model.load_state_dict(torch.load(
        model_save_name, map_location=torch.device(device)))
    model.to(device)
    model.eval()
    transform_test = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    combined = []
    print(f"test model {model_save_name}: \n {model}\n")
    for i in tqdm(range(1, 12501)):
        img_name = test_dataset_path+str(i)+'.jpg'
        img = Image.open(img_name)
        img_ = transform_test(img).unsqueeze(0)
        predictions = model(img_.to(device))

        label = torch.argmax(predictions).item()
        # label = torch.max(predictions, 1)[1].data.squeeze()
        combined.append([i, label])
    df = pd.DataFrame(combined, columns=['id', 'label'])
    df.to_csv(f"{model_save_name}_submission.csv", index=False)


def main():

    # 文件路径

    plat = platform.system().lower()
    if plat == 'windows':
        log_path = r"D:\Log"  # windows tensorbroad不支持中文路径
    elif plat == 'linux':
        log_path = r"./logs"  # 日志文件路径
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    # os.system(f"tensorboard --logdir={log_path} --bind_all")
    # $tensorboard --logdir="D:\Log" --bind_all 
    # $tensorboard --logdir="logs" --bind_all

    model_save_path = r"./model"  # 模型待存储路径
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    dataset_path = r"./data"  # 数据集存放路径

    # argparse
    parser = argparse.ArgumentParser(description='Dogs VS. Cats MYZHIBEI')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--seed', type=int, default=2023,
                        help='random seed (default: 1)')
    parser.add_argument('--workers', type=int, default=8,
                        help='multiworkers (default: 1)')

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)

    # dataset
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    train_dataset_path = dataset_path + '/train1/'
    val_dataset_path = dataset_path + '/validation/'

    train_dataset = ImageFolder(train_dataset_path, transform=transform)
    val_dataset = ImageFolder(root=val_dataset_path, transform=transform)

    # dataloader
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    vld_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False)

    n = 4
    dataiter = iter(train_loader)
    images, labels = dataiter.__next__()
    img_grid = torchvision.utils.make_grid(images[:n])
    imshow(img_grid)
    print(' '.join('%5s' % classes[labels[j].item()] for j in range(n)))
    writer.add_image('CATs OR DOGs Images', img_grid)

    writer.flush()
    model = Net(num_classes=2)

    # state_dict = torch.utils.model_zoo.load_url(
    #     'http://download.pytorch.org/models/alexnet-owt-7be5be79.pth')
    # for key in list(state_dict.keys()):
    #     if "classifier.6" in key:
    #         del state_dict[key]
    # model.load_state_dict(state_dict, strict=False)

    model.to(device)

    # optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,  lr_lambda=lambda epoch: 1/(epoch+1))

    # train
    save_dict = {"max": 0, "epoch": 0, "state_dict": model.state_dict()}
    for epoch in range(args.epochs):
        train(train_loader, model, loss_fn, optimizer, epoch, writer)        
        validate(vld_loader, model, loss_fn, epoch, writer, save_dict)
        scheduler.step()
    writer.close()

    time = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
    model_save_name = f"{model_save_path}/epoch_{save_dict['epoch']}_{device}_{time}.pth"

    torch.save(save_dict["state_dict"], model_save_name)

    test_dataset_path = dataset_path + '/test1/'
    test_model(model_save_name, test_dataset_path)
    


if __name__ == "__main__":
    main()
    # test_dataset_path = 'data/test1/'
    # model_save_name = 'model/epoch_1_cuda_20230528-160807.pth'
    # test_model(model_save_name, test_dataset_path)
    print("Finished")
