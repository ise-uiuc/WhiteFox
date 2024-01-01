
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(4)
        self.conv1 = torch.nn.Conv2d(3, 149, 3, stride=1, padding=0, dilation=1)
    def forward(self, x1):
        v1 = torch.relu(self.bn1(self.conv1(x1)))
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
