
import torchvision

class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv = torchvision.models.vgg16_bn(pretrained=True).features[40]
    def forward(self, x):
        v1 = self.deconv(x)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x = torch.randn(1, 64, 65, 65)
