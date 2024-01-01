
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, 3)
        self.bn1 = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(3, 3, 3)
        self.bn2 = torch.nn.BatchNorm2d(3)
        self.pool = torch.nn.AvgPool2d(5)
    def forward(self, x4):
        x4 = self.conv1(x4)
        x4 = self.relu(x4 + self.bn1(x4))
        x4 = self.conv2(x4)
        return torch.relu(x4 + self.bn2(x4))
# Inputs to the model
x4 = torch.randn(1, 3, 5, 5)
