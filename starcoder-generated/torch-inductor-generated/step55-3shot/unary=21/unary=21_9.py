
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1)
        self.conv2 = torch.nn.Conv2d(16, 64, 3)
        self.conv3 = torch.nn.Conv2d(64, 128, 5)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = torch.tanh(v1)
        v3 = self.conv2(v2)
        v4 = torch.tanh(v3)
        v5 = self.conv3(v4)
        v6 = torch.tanh(v5)
        return v6
# Inputs to the model
x = torch.randn(1, 1, 300, 300)
