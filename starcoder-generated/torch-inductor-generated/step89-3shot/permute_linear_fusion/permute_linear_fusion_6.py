
from torchvision import models
model = models.resnet18(pretrained = False)
# Inputs to the model
x1 = torch.randn([1, 3, 224, 244])
