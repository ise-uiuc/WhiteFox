
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 12, kernel_size=3)
    def forward(self, x):
        v1 = torch.tanh(x)
        self.conv(x)
        v2 = torch.tanh(v1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x = torch.randn(128, 16, 16, 16)
