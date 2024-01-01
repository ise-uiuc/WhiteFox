
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 11, stride=1, padding=5)
    def forward(self, x1, x2):
        v1 = torch.tanh(x1)
        v2 = self.conv(v1)
        v3 = v2 + x2
        v4 = torch.exp(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 28, 28)
x2 = torch.randn(1, 3, 28, 28)
