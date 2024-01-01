
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp_h, inp_c=None):
        v1 = torch.mm(x2, inp_h.transpose(-1, -2))
        v2 = x1 + v1.transpose(-1, -2)
        if inp_c is None:
            return v2
        else:
            return v2, v1.transpose(-1, -2)
# Inputs to the model
x1 = torch.randn(6, 48)
x2 = torch.randn(48, 10)
inp_h = torch.randn(6, 32)
inp_c = torch.randn(6, 32)
