
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, inp):
        v1 = torch.mm(x1, inp)
        v2 = torch.mm(v1, x1)
        return v2
# Inputs to the model
x1 = torch.randn(2, 2)
inp = torch.randn(2, 2)
