
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 6, bias=False)
    def forward(self, x1, x2, inp):
        v1 = self.linear(v2)
        v2 = v1.t() + x1
        v3 = torch.mm(v2, x2)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3, requires_grad=True)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
