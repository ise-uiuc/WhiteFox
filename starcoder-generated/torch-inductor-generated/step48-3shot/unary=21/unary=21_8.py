
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(47, 9, 1)
        self.conv1 = torch.nn.Conv2d(9, 1, 1)
        self.conv2 = torch.nn.Conv2d(1, 47, 1)
        self.conv3 = torch.nn.Conv2d(47, 1, 1)
        self.conv4 = torch.nn.Conv2d(1, 32, 3, padding=0)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv1(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv2(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv3(v6)
        v8 = torch.tanh(v7)
        v9 = self.conv4(v8)
        return v9
# Inputs to the model
x = torch.randn(1, 47, 56, 61)
