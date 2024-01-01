
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(16, 32, 3, stride=1, padding=1, groups=4)
        self.relu = torch.nn.ReLU(inplace=False)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.relu(v1)
        v3 = torch.add(v1, v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 16, 32, 32)
