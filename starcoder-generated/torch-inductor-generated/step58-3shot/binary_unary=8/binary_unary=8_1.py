
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(2, 8, 1, stride=1, padding=1)
    def forward(self, x1, x2, x3):
        v1 = torch.cat((x1, x2, x3), 1)
        v2 = self.conv1(v1)
        v3 = self.conv1(v1)
        v4 = v2 + v3
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 2, 64, 64)
x2 = torch.randn(1, 2, 64, 64)
x3 = torch.randn(1, 2, 64, 64)
