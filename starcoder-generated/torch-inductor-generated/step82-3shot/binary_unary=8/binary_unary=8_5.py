
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 8, 1, stride=2, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = torch.relu(v3)
        v5 = self.conv1(x1)
        v6 = v4 + v5
        return (torch.relu(v6))
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
