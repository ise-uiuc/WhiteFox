
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(8, 1, 2, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.tanh(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 8, 16, 16)
