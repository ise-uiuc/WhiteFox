
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 4, 3, stride=1, dilation=2)
    def forward(self, x1, x2):
        v1 = self.conv1(x1)
        v2 = self.conv1(x1)
        v3 = v1 + v2
        v4 = v1 + v2
        v5 = x2 + x1
        v6 = self.conv1(v5)
        v7 = v6 + v4
        v8 = torch.relu(v7)
        return v8
# Inputs to the model
x1 = torch.randn(1, 3, 256, 16)
x2 = torch.randn(1, 3, 256, 16)
