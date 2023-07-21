import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from networks import SmallCNNet
import wandb
import os
import pandas as pd
from torchvision.io import read_image
import argparse


def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, num_epochs=25):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                wandb.log({f"{phase}_accuracy": epoch_acc, f"{phase}_loss": epoch_loss})


                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
    return model





def init_transformation(image_size):
      
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def init_wandb(lr, model_arch, n_epochs):
    wandb.init(
    # set the wandb project where this run will be logged
    project="leaffliction",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": lr,
    "architecture": model_arch,
    "epochs": n_epochs,
    })

def init_model(model_arch, pretrained, n_classes, lr):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = 'IMAGENET1K_V1' if pretrained else None
    if model_arch == 'resnet':
        model_ft = models.resnet50(weights=weights)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)

    if model_arch == 'efficientnet':
        model_ft = models.efficientnet_b4(weights=weights)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_features=num_ftrs, out_features=n_classes)

    else:
        model_ft = SmallCNNet()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
    for param in model_ft.parameters():
        param.requires_grad = False
    for param in model_ft.classifier.parameters():
        param.requires_grad = True

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    return model_ft, criterion, optimizer_ft, exp_lr_scheduler

def init_dataset(data_transforms, data_dir):

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32,
                                                shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    return dataloaders, dataset_sizes, len(class_names)


def init(args):
    init_wandb(args['lr'], args['model_arch'], args['epochs'])
    tranform = init_transformation(args['image_size'])
    dataloaders, dataset_sizes, num_classes = init_dataset(tranform, args['dataset'])

    model, criterion, optimizer_ft, exp_lr_scheduler = init_model(args['model_arch'], args['pretrained'], num_classes, args['lr'])
    model = train_model(model, criterion, optimizer_ft, 
                        exp_lr_scheduler,
                         dataloaders,dataset_sizes,
                        num_epochs=args['epochs'])

    for param in model.parameters():
        param.require_grad = True
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model = train_model(model, criterion, optimizer_ft, 
                        exp_lr_scheduler,
                         dataloaders,dataset_sizes,
                        num_epochs=args['epochs'])

    torch.save(model.state_dict(), os.path.join(args['save_directory'], args['model_arch'] + '.pt'))

def create_dict(args):
    args_dict = {
        'image_size': args.image_size,
        'epochs': args.epochs,
        'pretrained': args.pretrained,
        'model_arch': args.model_arch,
        'save_directory': args.save_directory,
        'dataset': args.dataset,
        'lr': args.lr,

    }
    return args_dict

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='train',
                    description='Programme to train deep learning model')
    parser.add_argument('--dataset', '-d',  type=str, default=42, help='FOO!')    
    parser.add_argument('--save_directory', '-w',  type=str, default=42, help='FOO!')
    parser.add_argument('--model_arch', '-ma',  type=str, default=42, help='FOO!')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--image_size', '-s', type=int, default=224)
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--lr',type=float, default=0.001)
    args = parser.parse_args()
    return create_dict(args)

if __name__ == "__main__":
    args = parse_arguments()
    init(args)