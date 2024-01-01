
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 5, 1, stride=1)
    def forward(self, X0):
        v1 = self.conv1(X0)
        v2 = v1 - 63
        v3 = F.relu(v2)
        return v3
# Inputs to the model
X0 = torch.randn(1, 3, 64, 64)
