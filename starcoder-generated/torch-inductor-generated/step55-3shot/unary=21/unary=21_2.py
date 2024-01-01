
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=6, out_channels=8)
        self.conv2 = torch.nn.Conv2d(out_channels=16, in_channels=8, kernel_size=3, stride=3, padding=2)
        self.conv3 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
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
