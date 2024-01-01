
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2, 3))
    def forward(self, x1):
        x2 = self.weight * x1
        x3 = F.conv2d(x2, x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 2, 3, 3)
