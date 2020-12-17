# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 13:27:08 2020

@author: User
"""

import torch.nn as nn
from torchvision import models
from torch import nn

def set_parameter_requires_grad(model, freeze):
    for param in model.parameters():
        param.requires_grad = not freeze


def get_model(config):

    model = None
    # input_size = 0

    if config.model_name == "resnet":
        """ Resnet34
        """
        model = models.resnet34(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.fc.in_features
        model.fc = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "alexnet":
        """ Alexnet
        """
        model = models.alexnet(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "vgg":
        """ VGG16_bn
        """
        model = models.vgg16_bn(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    elif config.model_name == "densenet":
        """ Densenet
        """
        model = models.densenet121(pretrained=config.use_pretrained)
        set_parameter_requires_grad(model, config.freeze)

        n_features = model.classifier.in_features
        model.classifier = nn.Linear(n_features, config.n_classes)
        # input_size = 224
    else:
        raise NotImplementedError('You need to specify model name.')

    return model






# class Image_regression(nn.Module):
    
#     def __init__(self):
#         super(Image_regression, self).__init__()
        
#         self.layer = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding = 1),
#             nn.BatchNorm2d(16),
            
#             nn.Conv2d(16, 16, 3, padding = 1),
#             nn.BatchNorm2d(16),
#             nn.MaxPool2d(2),
            
            
#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.MaxPool2d(2),
            
#             nn.Conv2d(32, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
            
#             nn.Conv2d(64, 64, 3, padding = 1),
#             nn.BatchNorm2d(64),
#             nn.MaxPool2d(2)
#             )
        
#         self.fc = nn.Sequential(
#             nn.Linear(50176, 56),
#             nn.Linear(56, 1)
#         )
        
#     def forward(self, x):
#         out = self.layer(x)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         out = out.view(-1)
#         return out
