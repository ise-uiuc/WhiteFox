
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2):
        x1 = x1 + self.inp
        v1 = x1 + 2 * x2
        v2 = torch.mm(v1, x1)
        return v2 + x1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
