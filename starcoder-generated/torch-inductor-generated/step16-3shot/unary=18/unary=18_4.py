
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, dilation=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 + 1.0
        v3 = torch.tanh(v2)
        v4 = torch.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
