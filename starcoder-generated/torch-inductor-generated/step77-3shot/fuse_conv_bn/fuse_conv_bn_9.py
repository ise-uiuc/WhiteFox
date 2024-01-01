
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

# Model
def create_model(model_type="resnet18", pretrained=False):
    model = models.__dict__[model_type](pretrained=pretrained)
    return model

class Model(nn.Module):
    def __init__(self, model_def):
        super(Model, self).__init__()
        self.model = create_model(model_type=model_def)

    def forward(self, x):
        y = self.model(x)
        return y

# Input to the model
x1 = torch.randn(1, 3, 224, 224)
# Model
