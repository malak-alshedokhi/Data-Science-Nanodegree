
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
from torch import optim
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
from collections import OrderedDict

import argparse
parser = argparse.ArgumentParser(description='Train.py')
parser.add_argument('--data_dir', action = 'store', help = 'Enter path of data', type = str,default="flowers")
parser.add_argument('--save_dir', action = 'store', dest = 'save_directory', default = 'checkpoint.pth', help = 'where you want to save it')
parser.add_argument('--arch', action = 'store', dest = 'arch', type = str,  default = 'vgg16', help = 'Architecture of PreTrained model')
parser.add_argument('--learning_rate', action = 'store', dest = 'learning_rate',type = float, default = 0.001,help = 'Learning rate')
parser.add_argument('--hidden_units', action = 'store', dest = 'hidden_units',type = int,  default = 512, help = 'Number of hidden units')
parser.add_argument('--epochs', action='store', dest = 'epochs',type = int,  default = 1, help = 'How many epochs')
parser.add_argument('--gpu', action = 'store_true', dest = 'gpu',  default = False, help='Use GPU if --gpu')

pa = parser.parse_args()

data_dir = pa.data_dir
save_dir = pa.save_directory
learning_rate = pa.learning_rate
arch = pa.arch
hidden_units = pa.hidden_units
epochs = pa.epochs
gpu_mode = pa.gpu

  
def load_img(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(255),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle= True)
    validation_loader = torch.utils.data.DataLoader(validation_data, batch_size = 64)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size = 32)
    
    class_names = train_data.classes
    return class_names, train_loader, test_loader, validation_loader, train_data

def load_pretrained_model(model_name, hidden_units):
    model = getattr(models, model_name)(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    if 'vgg11' in model_name:
        input_size = model.classifier[0].in_features
    elif 'vgg16' in model_name:
        input_size = model.classifier[0].in_features
    else:
        print("Model not supported")
 
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, 512)),
                          ('relu', nn.ReLU()),
                          ('dropout' , nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units,200)),
                          ('relu', nn.ReLU()),
                          ('dropout' , nn.Dropout(0.2)),
                          ('fc3', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), learning_rate)
    return model, criterion, optimizer
def load_json(filename):
    
     return cat_to_name
def train_model(model, criterion, optimizer, lr,  trainloader, validloader, epochs, gpu = False):
    
    epochs = epochs
    print_every = 40
    steps = 0

    for e in range (epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate (trainloader):
            steps += 1
        if gpu is True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        optimizer.zero_grad ()
        outputs = model.forward (inputs)
        loss = criterion (outputs, labels)
        loss.backward ()
        optimizer.step ()

        running_loss += loss.item ()

        if steps % print_every == 0:
            model.eval ()


            with torch.no_grad():
                valid_loss, accuracy = validation(model, validloader, criterion)

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.4f}.. ".format(running_loss/print_every),
                  "Valid Loss: {:.4f}.. ".format(valid_loss/len(validloader)),
                  "Valid Accuracy: {:.4f}%".format(accuracy/len(validloader)*100))

            running_loss = 0
            model.train()

def validation(model, valid_loader, criterion, gpu = False):
    if gpu is True:
        model.to ('cuda')
    valid_loss = 0
    accuracy = 0
    for inputs, labels in valid_loader:
        if gpu is True:
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
        output = model.forward(inputs)
        valid_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return valid_loss, accuracy

def save_checkpoint(train_data, filename='checkpoint.pth'):
    model.class_to_idx = train_data.class_to_idx
   #creating dictionary
    checkpoint = {'arch' : arch,
                 'classifier': model.classifier,
                 'state_dict': model.state_dict(),
                 'mapping':    model.class_to_idx,
                 'Optimizer_dict': optimizer.state_dict(),
                 'epoch':1,
                 #'class_to_idx':model.class_to_idx,
                 'lr':0.001   }
    torch.save(checkpoint, filename ) 
if __name__== "__main__":
    print("Training has started...")
    class_names, train_loader, test_loader, validation_loader, train_data = load_img(data_dir)
    model, criterion, optimizer =  load_pretrained_model(arch, hidden_units)
    train_model(model, criterion, optimizer, learning_rate, train_loader, validation_loader, epochs, gpu_mode)
    save_checkpoint(train_data, save_dir)
    print('Training has ended...')
   

