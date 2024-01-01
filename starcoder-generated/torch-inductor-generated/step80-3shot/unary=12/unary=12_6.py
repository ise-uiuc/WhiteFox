
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(64, 16, 1, stride=1, padding=1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = torch.mul(self.conv(x1), self.sigmoid(self.conv(x1)))
        v2 = F.tanh(v1)
        v3 = torch.mul(self.conv(v2), F.relu(self.conv(v2)))
        v4 = v1 + v3
        return v4
# Inputs to the model
x1 = torch.randn(1, 64, 8, 100)
