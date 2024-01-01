
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x5):
        v1 = self.bn(x5)
        return v1
# Inputs to the model
x5 = torch.randn(1, 2, 48, 16, 24)
