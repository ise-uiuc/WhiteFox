
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.t = torch.randn(3, requires_grad=True)
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1 + self.t, -x2)
        return v1 + torch.sigmoid(inp)
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
