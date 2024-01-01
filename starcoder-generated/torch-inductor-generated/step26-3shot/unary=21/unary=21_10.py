
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 3, stride=2)
        self.conv2 = torch.nn.MaxPool2d(2)
        self.conv3 = torch.nn.Conv2d(32, 64, 3)
        self.conv4 = torch.nn.AvgPool2d(2)
        self.conv5 = torch.nn.Conv2d(64, 128, 3)
        self.conv6 = torch.nn.AvgPool2d(2)
    def forward(self, x):
        v1 = torch.tanh(self.conv(x))
        v2 = self.conv2(v1)
        v3 = torch.tanh(self.conv3(v2))
        v4 = self.conv4(v3)
        v5 = torch.tanh(self.conv5(v4))
        v6 = self.conv6(v5)
        return torch.tanh(v6)
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
