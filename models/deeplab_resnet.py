import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

def get_deeplabv3(num_classes=8, pretrained=True):
    model = deeplabv3_resnet50(weights="DEFAULT" if pretrained else None)
    model.classifier[-1] = nn.Conv2d(256, num_classes, 1)
    return model
