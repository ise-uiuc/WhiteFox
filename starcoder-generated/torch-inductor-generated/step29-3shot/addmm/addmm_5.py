
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x1)
        v2 = torch.zeros(3, 3, 3)
        v3 = torch.addcmul(v2, v1, inp, value=1.0)
        v4 = v3.mean()
        return v4
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, 3)
