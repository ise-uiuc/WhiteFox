
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.c2d1 = torch.nn.Conv1d(2, 3, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm1d(3, affine=False)
    def forward(self, x):
        x = self.c2d1(x)
        x = self.bn1(x)
        return x
# Inputs to the model
x = torch.randn(1, 2, 1, 10)
