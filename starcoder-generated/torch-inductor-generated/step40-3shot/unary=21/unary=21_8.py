
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(6, 6, 1)
        self.conv2 = torch.nn.Conv2d(6, 12, 1)
        self.conv3 = torch.nn.Conv2d(12, 9, 1)
        self.conv4 = torch.nn.Conv2d(9, 9, 3)
        self.conv5= torch.nn.Conv2d(9, 7, 3)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        v7 = self.conv4(v6)
        v8 = v7 + v1
        v9 = self.conv5(v8)
        return v9
# Inputs to the model
x = torch.randn(3, 6, 30, 10)
