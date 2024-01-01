
import torchvision
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool = torch.nn.MaxPool2d(1)
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
    def forward(self, x):
        v = self.maxpool(x)
        v1 = self.vgg16(v)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 224, 244)
