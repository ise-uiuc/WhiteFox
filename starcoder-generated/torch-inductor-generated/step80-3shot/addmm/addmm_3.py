
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3).expand(7, 3, 3)/2
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.inp
        return v2
# Inputs to the model
x1 = torch.randn(7, 3, 3)
x2 = torch.randn(7, 3, 3)
