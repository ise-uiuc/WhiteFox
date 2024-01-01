
class Model1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.randn(12, 6))
        self.w2 = torch.nn.Parameter(torch.randn(6, 6))
    def forward(self, x2, inp):
        v1 = torch.mm(x2, self.w1)
        v2 = v1 + inp
        return torch.mm(v2, self.w2)
# Inputs for the model
x2 = torch.randn(12, 6)
inp = torch.randn(6, 6)
