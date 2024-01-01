
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(70, 256, 1, stride=1)
        self.conv2 = torch.nn.Conv2d(35, 128, 3, stride=1, padding=1)
    def forward(self, X):
        v1 = self.conv1(X)
        v2 = v1 + 0.7948
        v3 = v1 - -0.3498
        v4 = v2 * v3
        v5 = v1 > -0.0806
        v6 = v5.float()
        v7 = v2 * v6
        v8 = v4 - v7
        v9 = torch.abs(v8)
        return v9
# Inputs to the model
X = torch.randn(1, 70, 1000, 1000)
