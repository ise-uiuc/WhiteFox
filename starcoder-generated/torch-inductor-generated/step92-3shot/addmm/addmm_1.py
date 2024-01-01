
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(3, 3))
    def forward(self, x, v0):
        v = torch.mm(self.weight.transpose(0, 1), x)
        return v + v0
# Inputs to the model
x = torch.randn(3, 3)
v0 = torch.randn(3, 3)
