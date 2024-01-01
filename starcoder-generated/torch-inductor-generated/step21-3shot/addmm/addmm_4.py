
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.sum(v1)
        v3 = v1
        v3 = v3 + inp
        return (v2, v3)
# Inputs to the model
x1 = torch.randn(3,4)
x2 = torch.randn(4,6)
inp = torch.randn(3,)
