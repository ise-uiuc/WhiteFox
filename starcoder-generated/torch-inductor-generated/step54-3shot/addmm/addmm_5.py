
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, inp)
        t3 = torch.mm(x2, x1)
        return t2 + t3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
