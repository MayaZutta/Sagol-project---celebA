import os
import torchvision
import torchvision.transforms as tfms
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from config import trainConfig
from torch.utils.tensorboard import SummaryWriter
# import torchmetrics
import torchvision.models as models
from clearml import Task
import datetime

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg_model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', weights=True)
vgg_model = models.vgg16(pretrained=True)


image_size = 224
imagenet_mean = [0.485, 0.456, 0.406]  # mean of the ImageNet dataset for normalizing
imagenet_std = [0.229, 0.224, 0.225]  # std of the ImageNet dataset for normalizing

transforms = tfms.Compose([tfms.Resize((image_size, image_size)),
                           tfms.ToTensor(),
                           tfms.Normalize(imagenet_mean, imagenet_std)])

def load_model():
    model = vgg_model
    model.classifier[6] = nn.Linear(in_features=4096, out_features=40, bias=True).eval()
    return model

def train_loop(data_dir, weights_dir, epochs=2):
    
    # Initialize SummaryWriter
    writer = SummaryWriter()
    
    train_dataset = torchvision.datasets.CelebA(data_dir, split="train", target_type=["attr"],
                                                transform=transforms, download=True)
    val_dataset = torchvision.datasets.CelebA(data_dir, split="valid", target_type=["attr"],
                                                transform=transforms, download=True)
    train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size= 16,
                              shuffle=True,
                              num_workers=4)
    val_dataloader = DataLoader(dataset=val_dataset,
                              batch_size=16,
                              shuffle=False,
                              num_workers=4)
    model = load_model()
    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,10,15], gamma=0.1)
    # acc=torchmetrics.Accuracy(task='multiclass', num_classes=config.num_classes).to(device)
    for epoch in tqdm(range(epochs)):

        model.train()
        epoch_train_loss, epoch_train_acc = [], []

        for i, data in enumerate(train_dataloader):
            # if i > 10:
                # break
            inputs = data[0]
            labels = data[1]
            print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            sigmoid_outputs = 1 / (1 + torch.exp(-outputs)).float()

            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            thresh_outputs = (sigmoid_outputs > 0.5).float() # MAYBE CHANGE IT TO 0.7
            curr_train_acc = np.mean(1-np.abs(labels-thresh_outputs).detach().numpy())

            # print statistics
            epoch_train_loss.append(loss.item())
            epoch_train_acc.append(curr_train_acc)
            if i % 1 == 0: # print every x mini-batches
                print(
                    f'[epoch: {epoch + 1}/{epochs},step: {i + 1:5d}/{len(train_dataloader)}] loss: {np.mean(epoch_train_loss):.3f}, acc: {np.mean(epoch_train_acc)}')
        scheduler.step()

        # SummaryWriter
        writer.add_scalar('Loss/train', np.mean(epoch_train_loss), epoch)
        writer.add_scalar('Accuracy/train', np.mean(np.array(epoch_train_acc)), epoch)
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

        # Model evaluation
        model.eval()
        epoch_val_loss, epoch_val_acc = [], []
        for i, data in enumerate(val_dataloader):
            # if i>10:
                # break
            inputs = data[0]
            labels = data[1]

            # forward
            outputs = model(inputs.float())
            # outputs = torch.nn.functional.softmax(outputs, dim=1)
            val_loss = criterion(outputs, labels.float())

            sigmoid_outputs = 1 / (1 + torch.exp(-outputs)).float()
            thresh_outputs = (sigmoid_outputs > 0.5).float()
            curr_val_acc = np.mean(1 - np.abs(labels - thresh_outputs).detach().numpy())
            epoch_val_loss.append(val_loss.item())
            epoch_val_acc.append(curr_val_acc)


        torch.save(model, f'{weights_dir}/epoch_{epoch + 1}_loss_{np.round(loss.detach().numpy(), decimals=3)}.pt')

        print(f'[epoch: {epoch + 1}/{epochs}] loss: {np.mean(epoch_val_loss):.3f}, acc: {np.mean(epoch_val_acc)}')
            
        # SummaryWriter
        writer.add_scalar('Loss/val', np.mean(epoch_val_loss), epoch)
        writer.add_scalar('Accuracy/val', np.mean(np.array(epoch_val_acc)), epoch)
        writer.close()


if __name__ == '__main__':
    # this statement (if __name__ == ...) is executed only if the script is run as the main program
    conf = trainConfig()
    root_dir = conf.root_dir
    data_dir = conf.data_dir
    exp_name = conf.exp_name
    exp_dir = f'{root_dir}/{exp_name}'
    weights_dir = f'{exp_dir}/weights'
    for dir in [exp_dir, weights_dir]:
        if not os.path.isdir(dir):
            os.mkdir(dir)
    current_time = datetime.datetime.now()
    task = Task.init(project_name='training CelebA', task_name=f'Task {current_time.strftime("%m%d_%H%M")}')
    logger = task.get_logger()
    train_loop(data_dir, weights_dir, conf.epochs)


