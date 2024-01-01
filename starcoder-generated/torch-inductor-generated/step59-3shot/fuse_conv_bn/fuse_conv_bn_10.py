
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(1)
    def forward(self, x):
        x = self.bn1(x)
        return x
# Inputs to the model
x = torch.randn(1, 1, 1, 2)
