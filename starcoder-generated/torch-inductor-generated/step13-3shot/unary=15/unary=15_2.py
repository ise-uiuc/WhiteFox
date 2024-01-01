
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (3, 18), stride=1, padding=0)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 128, (5, 1), stride=1, padding=0)
        self.bn2 = torch.nn.BatchNorm2d(128)
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.bn1(v1)
        v3 = torch.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.bn2(v4)
        v6 = torch.relu(v5)
        v7 = self.maxpool(v6)
        v8 = torch.flatten(v7, 1)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 128, 256)
