
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=3)
    def forward(self, x1):
        v1 = torch.nn.Sigmoid()
        v2 = v1(x1)
        v3 = self.conv(v2)
        v4 = torch.tanh(v3)
        v5 = torch.mul(v2, v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
