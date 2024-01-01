
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, inp):
        v1 = torch.mm(x2, torch.clone(inp))
        return torch.add(v1, x1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3, requires_grad=True)
