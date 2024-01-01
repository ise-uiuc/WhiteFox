
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.linear = torch.nn.Linear(16, 16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.linear(v1)
        v3 = self.linear(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
