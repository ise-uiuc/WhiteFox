
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 1, 8, stride=2, padding=1, dilation=2)
        self.conv2 = torch.nn.Conv2d(1, 27, 2, stride=2, padding=0, dilation=1)
        self.conv3 = torch.nn.Conv2d(27, 40, 4, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(40, 27, 4, stride=2, padding=1)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        return torch.tanh(v7)
# Inputs to the model
x = torch.randn(1, 3, 75, 75)
