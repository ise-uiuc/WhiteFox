
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
        self.inp2 = torch.randn(3, 3)
    def forward(self, x1):
        v1 = torch.mm(x1, x1)
        v4 = torch.mm(x1, x1)
        v2 = v1 + self.inp2
        v3 = v4
        return v3
# Inputs to the model
x1 = torch.randn(3, 3)
