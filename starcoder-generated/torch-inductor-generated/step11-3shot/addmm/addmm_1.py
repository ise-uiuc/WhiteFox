
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(0.0000039 / torch.exp2(x1), torch.abs(x2))
        v2 = -v1 + inp
        return v2
# Inputs to the model
x1 = torch.randn(1, 35)
x2 = torch.randn(35, 35)
inp = torch.randn(35, 35)
