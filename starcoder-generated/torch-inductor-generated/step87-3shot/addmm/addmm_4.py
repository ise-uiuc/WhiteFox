
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, inp):
        v1 = torch.mm(x1, x2)
        v1 = torch.mm(inp, x3)
        return inp + v1
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
x3 = torch.randn(4, 4)
inp = torch.randn(3, 3)
