
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(16)
    def forward(self, x):
        y1 = self.conv1(x)
        z1 = self.bn1(y1)
        a1 = torch.tanh(z1)
        return a1
# Inputs to the model
x = torch.rand(1, 3, 47, 63)
