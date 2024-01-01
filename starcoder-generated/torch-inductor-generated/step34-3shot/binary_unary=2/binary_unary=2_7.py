
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = torch.nn.MaxPool2d(2, stride=2, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1)
        self.maxpool2 = torch.nn.MaxPool2d(2, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=1)
    def forward(self, x1):
        v1 = self.maxpool1(x1)
        v2 = self.conv1(v1)
        v3 = v2 - 0.5
        v4 = F.relu(v3)
        v5 = self.maxpool2(v4)
        v6 = self.conv2(v5)
        v7 = v6 - 0.2
        v8 = F.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
