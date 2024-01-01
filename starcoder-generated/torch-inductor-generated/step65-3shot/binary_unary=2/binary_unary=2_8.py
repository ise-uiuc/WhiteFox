
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.gap = torch.nn.AdaptiveAvgPool2d(output_size = 1)
    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.gap(v1)
        v3 = v2/0.8
        v4 = F.relu(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
