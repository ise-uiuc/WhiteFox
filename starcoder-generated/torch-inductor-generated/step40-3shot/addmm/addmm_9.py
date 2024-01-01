
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v = torch.sigmoid(torch.mm(x1, inp) / 0.5)
        v2 = torch.mm(x2, v)
        return v + v2, x1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
