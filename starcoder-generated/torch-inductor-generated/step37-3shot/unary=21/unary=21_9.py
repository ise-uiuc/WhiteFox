
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 56, 1)
    def forward(self, x):
        v1 = self.conv(x)
        return torch.atanh(v1)
# Inputs to the model
x = torch.randn(1, 64, 1, 1)
