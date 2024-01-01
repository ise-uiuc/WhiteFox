
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randint(10, (3, 3)))
    def forward(self, x1, x2, inp):
        v1 = torch.mm(self.weight, x1)
        v2 = v1 + torch.nn.functional.dropout(inp)
        v3 = torch.mm(x2, v2)
        out = torch.mm(x2, self.weight.t())
        out = torch.mm(x2, torch.nn.functional.relu(inp))
        return out
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3, requires_grad=True)
inp = torch.randn(3, 3)
