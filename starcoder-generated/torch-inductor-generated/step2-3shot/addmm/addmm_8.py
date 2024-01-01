
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.input1 = torch.rand(1, 2)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.input1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(6, 12)
x2 = torch.randn(12, 6)
inp = torch.randn(6, 6)
