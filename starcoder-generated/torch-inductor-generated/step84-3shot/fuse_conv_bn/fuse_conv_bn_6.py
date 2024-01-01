
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(8)
    def forward(self, x1):
        v1 = self.bn(x1 + 4)
        return torch.relu(v1 *.2)
# Inputs to the model
x1 = torch.randn(2, 8)
