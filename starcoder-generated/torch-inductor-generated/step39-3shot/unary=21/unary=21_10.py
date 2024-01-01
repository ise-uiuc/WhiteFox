
class ModelTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 50, 2, padding=0, groups=2, bias=False)
    def forward(self, x0):
        v1 = self.conv(x0)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x0 = torch.randn(10, 1, 36, 24)
