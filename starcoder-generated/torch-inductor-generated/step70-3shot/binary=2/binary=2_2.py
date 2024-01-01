
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = torch.nn.Conv2d(1, 20, (5, 5))
        self.c2 = torch.nn.Conv2d(20, 20, (1, 1))
    def forward(self, x):
        v0 = self.c1(x)
        v1 = self.c2(v0)
        v2 = v1 - 10.0
        return (v0, v2)
# Inputs to the model
x = torch.rand(1, 1, 28, 28)
