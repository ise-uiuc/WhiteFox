
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.relu2 = torch.nn.ReLU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        return self.relu2(self.bn2(x))
# Inputs to the model
x = torch.randn(1, 3, 480, 640)
