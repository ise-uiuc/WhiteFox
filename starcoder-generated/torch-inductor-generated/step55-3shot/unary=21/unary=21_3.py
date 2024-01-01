
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 3, 1, 1)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0, dilation=1, ceil_mode=False)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.maxpool2 = torch.nn.MaxPool2d(3, 3, 0, 1, False)
        self.relu1 = torch.nn.ReLU()
    def forward(self, x4):
        v1 = self.conv1(x4)
        v2 = self.maxpool1(v1)
        v3 = self.relu(v2)
        v4 = self.conv2(v3)
        v5 = self.maxpool2(v4)
        v6 = self.relu1(v5)
        return v6
# Inputs to the model
x4 = torch.randn(16, 3, 112, 112)
