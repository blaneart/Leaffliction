import PySimpleGUI as sg
import cv2
from networks import SmallCNNet
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus as GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import argparse
import os


def init_transforms(image_size):
    transf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transf


class Model(nn.Module):
    def __init__(self, baseline, n_classes):
        super(Model, self).__init__()
        self.baseline = baseline
        self.fc = nn.Sequential(nn.Linear(1000, 512), nn.LeakyReLU(),
                                nn.Linear(512, n_classes))

    def forward(self, x):
        x = self.baseline(x)
        return self.fc(x)


def init_model(model_arch, n_classes, weights, image_size, device):
    transf = init_transforms(image_size)
    if model_arch == 'resnet':
        model_ft = models.resnet50()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
        target_layers = [model_ft.layer3[-1]]
    elif model_arch == 'efficientnet_v2':
        model_ft = models.efficientnet_v2_s(weights=weights)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = nn.Linear(in_features=num_ftrs,
                                           out_features=n_classes)
        target_layers = [model_ft.features[-1][0]]
    elif model_arch == 'efficientnet':
        model_ft = Model(models.efficientnet_b4(weights=None), n_classes)
        target_layers = [model_ft.baseline.features[-1][0]]
    else:
        model_ft = SmallCNNet()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, n_classes)
        target_layers = [model_ft.conv2]
    model_ft = model_ft.to(device)
    saved_dict = torch.load(weights,
                            map_location=torch.device('cpu'))
    model_ft.load_state_dict(saved_dict['model'])
    classes = saved_dict['labels']
    for param in model_ft.parameters():
        param.requires_grad = True
    return model_ft, transf, target_layers, classes


def gradcam(model_ft, target_layers, transf, path, device):
    cam = GradCAM(model=model_ft, target_layers=target_layers,  use_cuda=False)
    image = Image.open(path).convert('RGB')
    transformed = transf(image)
    pred = model_ft(transformed.unsqueeze(0).to(device))
    grayscale_cam = cam(input_tensor=transf(image).unsqueeze(0))
    im = image.resize((224, 224)).convert("RGB")
    rgb_img = np.array(im)
    rgb_img = np.float32(rgb_img) / 255
    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam)
    return pred, transformed, visualization


def app(image, grad_image, predicted_class):
    layout = [
        [sg.Text('Original'),
         sg.Text('GradCam')],
        [sg.Image(data=image, key='image'),
         sg.Image(data=grad_image, key='grad_image')],
        [sg.Text('===  DL classification  ===')],
        [sg.Text('Update Image'), sg.In(r'image', size=(40, 1), key='update'),
         sg.FileBrowse()],
        [sg.Text(predicted_class, key='prediction')],

        [sg.OK('Update'), sg.Button('Quit')]]

    window = sg.Window('Prediction',
                       default_element_size=(14, 1),
                       text_justification='center',
                       auto_size_text=True).Layout(layout)
    return window


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
                        default='./weights/apple_weights/efficientnet_v2.pt')
    args = parser.parse_args()
    return create_dict(args)


def run(window, model_ft, target_layers, transf, classes, device):
    while True:
        event, values = window.read()
        if os.path.isfile(values['update']):
            img = cv2.imencode('.png',
                               cv2.resize(cv2.imread(values['update']),
                                                    (256, 256)))[1].tobytes()
            pred, transformed, visualization = gradcam(model_ft,
                                                       target_layers,
                                                       transf,
                                                       values['update'],
                                                       device)
            visualization = cv2.imencode('.png', visualization)[1].tobytes()
            window['image'].update(data=img)
            window['grad_image'].update(data=visualization)
            window['prediction'].update(classes[torch.argmax(pred, 1)])
            window['image'].update(data=img)
    # See if user wants to quit or window was closed
        if event == sg.WINDOW_CLOSED or event == 'Quit':
            break  # Output a message to the window
        print(values['update'])
    window.close()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = cv2.resize(cv2.imread(args['image'][0]), (256, 256))
    model_ft, transf, target_layers, classes = init_model(args['model_arch'],
                                                          4,
                                                          args['weights'],
                                                          args['image_size'],
                                                          device)

    pred, transformed, visualization = gradcam(model_ft, target_layers,
                                               transf, args['image'][0],
                                               device)
    predicted_class = classes[torch.argmax(pred, 1)]

    imgbytes = cv2.imencode('.png', image)[1].tobytes()
    visualization = cv2.imencode('.png', visualization)[1].tobytes()
    transformed = cv2.imencode('.png', image)[1].tobytes()
    sg.theme('light green')
    window = app(imgbytes, visualization, predicted_class)
    run(window, model_ft, target_layers, transf, classes, device)
