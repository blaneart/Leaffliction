import torch
import time
from torchvision import datasets, models, transforms
from .train import init_model
import os
import torch
import torch.nn as nn
import argparse

def init_dataset(data_transforms, data_dir):
    image_datasets = datasets.ImageFolder(os.path.join(data_dir),
                                            data_transforms)
    dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=32, num_workers=4)
    dataset_sizes = len(image_datasets)
    class_names = image_datasets.classes
    return dataloaders, dataset_sizes, class_names


def eval_model(model, criterion, dataloaders, dataset_sizes):
    since = time.time()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()   # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders:
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        # forward
        # track history if only in train
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / dataset_sizes
    epoch_acc = running_corrects.double() / dataset_sizes

    print(f'Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    time_elapsed = time.time() - since
    print(f'Evaluation complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')


def init_transformation(image_size):
      
    data_transforms = {
        'val': transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def init(args):
    tranform = init_transformation(args['image_size'])
    dataloaders, dataset_sizes, classes = init_dataset(tranform, args['dataset'])
    model_ft, criterion, _, _ = init_model(args['model_arch'], False, len(classes), 0.01)
    model_ft.load_state_dict(torch.load(args['weights']['model']))
    eval_model(model_ft, criterion, dataloaders, dataset_sizes)

def create_dict(args):
    args_dict = {
        'image_size': args.image_size,
        'model_arch': args.model_arch,
        'dataset': args.dataset,
        'weights': args.weights,
    }
    return args_dict

def parse_arguments():
    parser = argparse.ArgumentParser(
                    prog='eval',
                    description='Programme to train deep learning model')
    parser.add_argument('--dataset', '-d',  type=str, default=42, help='FOO!')    
    parser.add_argument('--model_arch', '-ma',  type=str, default=42, help='FOO!')
    parser.add_argument('--image_size', '-s', type=int, default=224)
    parser.add_argument('--weights', '-w', type=int, default=5)
    args = parser.parse_args()
    return create_dict(args)


if __name__ == "__main__":
    args = parse_arguments()
    init(args)