
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 6, 1)
        self.bn1 = torch.nn.BatchNorm2d(6)
        self.conv2 = torch.nn.Conv2d(6, 6, 1)
        self.bn2 = torch.nn.BatchNorm2d(6)
        self.conv3 = torch.nn.Conv2d(6, 6, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        return x
# Inputs to the model
x = torch.randn(1, 6, 6, 6)
