
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
        self.inp3 = torch.randn(3, 3)
        self.inp4 = torch.randn(3, 3)
        self.inp5 = torch.randn(3, 3)
    def forward(self, x1):
        v1 = torch.mm(x1, x1)
        v2 = self.inp3
        v3 = torch.mm(x1, v2)
        v4 = torch.mm(x1, x1)
        v5 = v4 + x1
        v6 = torch.mm(x1, x1)
        return v3
# Inputs to the model
x1 = torch.randn(3, 3)
