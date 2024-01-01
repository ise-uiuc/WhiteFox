
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
        self.inp2 = torch.randn(3, 3)
    def forward(self, x1, x2, x3):
        v1 = torch.mm(x1, x1) + self.inp1
        v2 = torch.mm(x2, x2)
        v3 = torch.mm(x3, x3)
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
