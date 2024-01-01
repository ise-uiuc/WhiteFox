
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super(ModelTanh, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 6, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(6, 9, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.bn2 = torch.nn.BatchNorm2d(6)
        self.bn3 = torch.nn.BatchNorm2d(9)
        self.tanh = torch.nn.Tanh()
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        x = self.tanh(x)
        return x
# Inputs to the model
x = torch.randn(1, 3, 32, 32)
