
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv1d(3, 5, 1, 1)
        self.bn = torch.nn.BatchNorm1d(num_features=2, affine=True)
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(y)
        return y
# Inputs to the model
x = torch.rand(2, 1, 4)
x = x.expand(2, 2, 4)
