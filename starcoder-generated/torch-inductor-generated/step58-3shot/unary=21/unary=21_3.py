
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu1 = torch.nn.ReLU()
        self.conv = torch.nn.Conv2d(29, 29, (20, 20), stride=[1, 1], padding=[0, 0], dilation=[1, 1], groups=29, bias=True)
        self.conv2 = torch.nn.Conv2d(29, 29, (1, 1), stride=1, padding=0, dilation=1, groups=29, bias=True)
    def forward(self, x):
        v1 = self.relu1(x)
        v2 = self.conv(v1)
        v3 = torch.tanh(v2)
        v4 = self.conv2(v3)
        return self.tanh(v4)
# Inputs to the model
x = torch.randn(39, 29, 51, 91)
