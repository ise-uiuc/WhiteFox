
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
    def forward(self, x1, other, inplace=True):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v2 = v2 + other
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
other = 1
