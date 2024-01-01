
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 17, 3, stride=2, padding=0)
    def forward(self, X):
        v1 = self.conv(X)
        v2 = v1 - 2.0
        v3 = F.relu(v2)
        return v3
# Input to the model
X = torch.randn(1, 4, 1000, 1000)
