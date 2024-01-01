
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v  = x1 + x2
        v1 = torch.mm(v, inp)
        v2 = v1 - x1
        v3 = torch.mm(v, inp)
        # The output should be equal to the output of the next two lines of code.
        v4 = torch.mm(torch.matmul(x1, x2), inp)
        v5 = torch.mm(torch.matmul(x1, x2), inp)
        return v3 + inp
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad = True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
