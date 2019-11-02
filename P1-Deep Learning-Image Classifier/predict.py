import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import json
import PIL
from PIL import Image
import argparse


parser = argparse.ArgumentParser(description='Predict.py')
parser.add_argument('input_img', default='./flowers/test/1/image_06752.jpg', nargs='*', action="store", type = str)
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")
parser.add_argument('--checkpoint', default='./checkpoint.pth', nargs='*', action="store",type = str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--arch', action = 'store', dest = 'arch', type = str,  default = 'vgg16', help = 'Architecture of PreTrained model')

pa = parser.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint
model_name = pa.arch
category_names = pa.category_names

#1
def load_pretrained_model(model_name, hidden_units):
    model = models.model_name(pretrained=True)
    for param in model.parameters():
     param.requires_grad = False  
 
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 500)),
                          ('relu', nn.ReLU()),
                          ('dropout' , nn.Dropout(0.2)),
                          ('fc2', nn.Linear(500, 200)),
                          ('relu', nn.ReLU()),
                          ('dropout' , nn.Dropout(0.2)),
                          ('fc3', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    return model

#2
def loading_checkpoint(file_path):
    checkpoint = torch.load(file_path) 
    if checkpoint['arch'] == 'vgg16':
       model2=models.vgg16(pretrained=True)
    elif checkpoint['arch'] == 'densenet161':
        model2=models.densenet161(pretrained=True)
    else:
        print('Please choose your pretrained model..')
        model2=models.vgg16(pretrained=True)
    input_size = 25088
    hiddin_size = [3000 , 500]
    output_size = 102
   # hidden_units = checkpoint['hidden_units']
              
    for param in model2.parameters(): 
        param.requires_grad = False #turning off tuning of the model
    model2.classifier= checkpoint['classifier']
    model2.class_to_idx = checkpoint['mapping']
    model2.optimizer_dict = checkpoint['Optimizer_dict']
    model2.load_state_dict(checkpoint['state_dict'])
    lr = checkpoint['lr']
    epoch=checkpoint['epoch']
    
    return model2
#3
def process_image(image):
    img_pil = Image.open(image)
   
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor
#4
def predict(image_path, model_name, topk=5 , gpu = False):
    model.to('cuda:0')
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    with torch.no_grad():
        output = model.forward(img_torch.cuda())
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)

if __name__== "__main__":
    print("Prediction has started...")
    model = loading_checkpoint(path)
    probabilities = predict(path_image, model, number_of_outputs, power)
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
    a = np.array(probabilities[0][0])
    b = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
    print(a)
    print(b)
   
    print("Prediction has ended..")
