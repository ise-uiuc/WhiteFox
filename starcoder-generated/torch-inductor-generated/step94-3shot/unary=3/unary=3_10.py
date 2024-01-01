
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 24, 3, stride=1, padding=1)
        self.relu = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.relu(v1)
        v3 = v2 * 0.5
        v4 = v2 * 0.6931471805599453
        v5 = torch.erf(v4)
        v6 = v5 + 1
        v7 = v3 * v6
        return v7
# Inputs to the model
x1 = torch.randn(1, 32, 7, 7)
