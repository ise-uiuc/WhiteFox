
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 3, 2)
        self.bn = torch.nn.BatchNorm1d(3, affine=False)
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 1, 8)
