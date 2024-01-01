
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x4, x5, inp):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x4, x5)
        v3 = torch.mm(v1, v2)
        v4 = v3 + inp
        return v4
# Inputs to the model
# Inputs begin
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3, requires_grad=False)
x4 = torch.randn(3, 3, requires_grad=False)
x5 = torch.randn(3, 3)
inp = torch.randn(3, 3, 3)
# Inputs end
