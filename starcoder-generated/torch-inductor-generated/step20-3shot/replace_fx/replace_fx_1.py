
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Conv2d(1, 1, 1, bias=False)
    def forward(self, x1):
        x2 = self.m1(x1)
        x3 = torch.rand_like(x2)
        return x3
# Inputs to the model
x1 = torch.randn(1, 1, 2, 2)
