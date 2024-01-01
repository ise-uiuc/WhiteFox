
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, inp)
        v2 = v1.permute(1, 0)
        t1 = v1 + v2
        v3 = t1.permute(1, 0)
        return torch.mm(v3, inp)
# Inputs to the model
x1 = torch.randn(4, 12)
x2 = torch.randn(12, 4)
inp = torch.randn(4, 4)
