
from torchvision.models import LeNet
class Model(LeNet):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        y = x1
        v1 = torch.nn.functional.linear(y, self.layer4.weight, self.layer4.bias)
        v2 = v1.permute(0, 2, 1)
        return v1
# Inputs to the model
x1 = torch.randn(3, 1, 32, 32)
