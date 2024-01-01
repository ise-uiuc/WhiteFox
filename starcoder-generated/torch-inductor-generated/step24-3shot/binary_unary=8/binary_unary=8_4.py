
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 8, 2, stride=2, padding=0)
        self.linear = torch.nn.Linear(8, 16)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv(v1)
        v3 = v2.flatten(start_dim=1)
        v4 = self.linear(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
