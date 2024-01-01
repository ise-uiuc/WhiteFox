
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sin(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v2)
        v5 = torch.clamp(v2, min=0.0, max=6.0)
        return v5
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
