"""
Baseline UNet model for Ice-Bench segmentaiton
UNet Resnet50, DeepLabv3
"""



# Import libraries
import torch
import segmentation_models_pytorch as smp



class UNet:
    def __init__(self):
        pass




# models




model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)