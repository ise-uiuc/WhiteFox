
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 24, 1)
    def forward(self, x):
        v1 = self.conv(x)
        v3 = v1.flatten(1)
        v2 = torch.tanh(v3)
        return v2
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
