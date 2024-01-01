
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(19, 49, 1, dilation=3, padding=3)
        self.sigmoid = torch.nn.Sigmoid()
        self.conv1 = torch.nn.Conv2d(49, 19, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(19, 1, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.sigmoid(v1)
        v3 = self.conv1(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        return v5
# Inputs to the model
x = torch.randn(49, 19, 58, 56)
