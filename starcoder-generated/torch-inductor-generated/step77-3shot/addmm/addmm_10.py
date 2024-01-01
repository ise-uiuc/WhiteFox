
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, x1, x2):
        v1 = torch.mm(x1, self.inp)
        v2 = torch.add(v1, x1, alpha=1)
        v3 = torch.mm(v2, x2)
        return v3
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 2)
