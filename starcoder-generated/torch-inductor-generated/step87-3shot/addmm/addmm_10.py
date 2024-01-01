
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = torch.nn.ModuleList([torch.nn.Linear(3, 3)]) # initialize as a list
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x1, x2) + inp
        v2 = torch.mm(inp, inp)
        v3 = torch.mm(x1, v2) + self.mm[0]
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
