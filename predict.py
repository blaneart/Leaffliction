from networks import SmallCNNet
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import argparse


def init_transforms(image_size):
    transf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transf


def init_model(model_arch, n_classes, weights, image_size, img_path):
    transf = init_transforms(image_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_arch == 'resnet':
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
        target_layers = [model_ft.layer3[-1]]

    if model_arch == 'efficientnet':
        model_ft = models.efficientnet_b4()
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_features=num_ftrs,
                                           out_features=n_classes)
        target_layers = [model_ft.features[-2][0].block[-1][0]]
        print(target_layers)

    else:
        model_ft = SmallCNNet()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
        target_layers = [model_ft.conv2]
    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(weights))
    # model_ft.eval()
    for param in model_ft.parameters():
        param.requires_grad = True
    gradcam(model_ft, target_layers, transf, img_path, device)


def gradcam(model_ft, target_layers, transf, path, device):
    cam = GradCAM(model=model_ft, target_layers=target_layers,  use_cuda=True)
    image = Image.open(path)
    image.save('kek2.jpg')
    print(model_ft(transf(image).unsqueeze(0).to(device)))

    grayscale_cam = cam(input_tensor=transf(image).unsqueeze(0))
    im = image.resize((224, 224)).convert("RGB")
    rgb_img = np.array(im)
    rgb_img = np.float32(rgb_img) / 255
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    im = Image.fromarray(visualization).save('kek.jpg')


def create_dict(args):
    args_dict = {
        'image_size': args.image_size,
        'model_arch': args.model_arch,
        'weights': args.weights,
        'image': args.image,
    }
    return args_dict


def parse_args():
    parser = argparse.ArgumentParser(
                    prog='predict',
                    description='Programme to predict class')
    parser.add_argument('image', metavar='image', type=str, nargs=1,
                        help='directory to augment'
                        'information about images from')
    parser.add_argument('--model_arch', '-ma',  type=str,
                        default='efficientnet', help='FOO!')
    parser.add_argument('--image_size', '-s', type=int, default=224)
    parser.add_argument('--weights', '-w', type=str,
                        default='./models/efficientnet.pt')
    args = parser.parse_args()
    return create_dict(args)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    init_model(args['model_arch'], 4, args['weights'],
               args['image_size'], args['image'][0])
