
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 16, [1, 3], stride=1, padding=1)
        self.linear = torch.nn.Linear(432, 432)
    def forward(self, x1):
        v1 = self.conv(x1)
        v3 = torch.flatten(v1, 1)
        v2 = self.linear(v3)
        v4 = torch.softmax(v2, dim=0)
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
