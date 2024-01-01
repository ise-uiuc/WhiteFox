
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1.t(), inp)
        v2 = v1 + x2
        return v2
# Inputs to the model
x1 = torch.randn(1, 35)
x2 = torch.randn(16, 35)
inp = torch.randn(32, 35)
