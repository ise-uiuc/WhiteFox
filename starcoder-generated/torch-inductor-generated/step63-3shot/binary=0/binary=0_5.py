
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(5, 1)
    def forward(self, x1, x2, other=1):
        v1 = self.conv(x1)
        v2 = self.linear(x2)
        v3 = v1 + v2
        v4 = v3 + other
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 5)
