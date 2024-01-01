
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c = torch.nn.Conv3d(1, 2, 1)
        self.bn = torch.nn.BatchNorm3d(2)
    def forward(self, x1):
        return self.c(self.bn(x1))
# Inputs to the model
x1 = torch.randn(4, 1, 4, 4, 4)
