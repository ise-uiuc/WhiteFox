
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 5, 3)
        self.bn = torch.nn.BatchNorm1d(6, affine=False)
    def forward(self, x1):
        y = self.bn(self.conv(x1))
        return y
# Inputs to the model
x1 = torch.randn(1, 3, 4, 4)
