
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, inp)
        if torch.equal(torch.add(inp, v1), torch.reshape(torch.add(inp, v1), (3, 3))):
            return v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3, requires_grad=True)
