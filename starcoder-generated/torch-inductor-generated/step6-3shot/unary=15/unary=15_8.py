
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 32, padding=8, dilation=24)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 128, 128)
