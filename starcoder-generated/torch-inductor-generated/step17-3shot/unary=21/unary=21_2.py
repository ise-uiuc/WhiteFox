
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 168, 21, groups=11, padding=3, stride=2)
        self.conv2 = torch.nn.Conv2d(168, 64, 11, groups=9, padding=0, stride=1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = torch.tanh(v2)
        v3 = self.conv2(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 3, 121, 73)
