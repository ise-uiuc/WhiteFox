
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=8, kernel_size=3)
        self.conv2 = torch.nn.Conv2d(8, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 32, 3)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 6, 224, 224)
