
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2)
        v2 = v1 + torch.tanh(inp)
        return v2
# Inputs to the model
x1 = torch.rand(1, 2, 3, 3, requires_grad=True)
x2 = torch.rand(1, 2, 3, 3, requires_grad=True)
inp = torch.rand(1, 2, 3, 3, requires_grad=True)
