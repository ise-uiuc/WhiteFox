
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 24, 1, stride=1, padding=1)
        self.linear = torch.nn.Linear(16, 1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.linear(x1)
        return v3
# Inputs to the model
x1 = torch.randn(1, 2, 3, 4)
