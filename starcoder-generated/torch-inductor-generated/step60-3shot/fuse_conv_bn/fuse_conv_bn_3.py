
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(1, 4, 1, 2)
        self.bn = torch.nn.BatchNorm1d(4, momentum=0.0, affine=True)
    def forward(self, x):
        return self.bn(self.conv(x))
# Inputs to the model
x = torch.randn(1, 1, 2)
