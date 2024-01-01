
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=2, padding=1)
    def forward(self, x0):
        v0 = self.conv(x0)
        v1 = torch.tanh(v0)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
