
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(9, 16, 7, stride=1, padding=3)
        self.linear = torch.nn.Linear(16, 2)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        v1 = self.conv(x)
        v2 = self.linear(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
