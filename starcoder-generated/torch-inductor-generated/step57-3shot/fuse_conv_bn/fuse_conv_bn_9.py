
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(16, 33, 3)
        self.bn = torch.nn.BatchNorm1d(33, affine=False)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
# Inputs to the model
x = torch.randn(1, 16, 14)
