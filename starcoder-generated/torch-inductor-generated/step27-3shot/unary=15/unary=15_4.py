
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(64)
    def forward(self, x1):
        v1 = self.bn(x1)
        v2 = F.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 64, 64, 64)
