
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False, dilation=1, groups=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5 # Add a model specific tensor or scalar
        v3 = F.relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 16, 128, 128)
