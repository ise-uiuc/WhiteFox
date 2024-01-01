
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t1 = torch.nn.Conv2d(1, 16, 1)
        self.t2 = torch.nn.Conv2d(16, 16, 1)
    def forward(self, x1):
        v1 = self.t1(x1)
        v2 = self.t2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
