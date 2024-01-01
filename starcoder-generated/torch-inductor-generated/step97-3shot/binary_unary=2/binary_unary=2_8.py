
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 - 1.6
        v3 = F.tanh(v2)
        v4 = self.conv2(x1)
        v5 = v4 - 0.75
        v6 = F.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
