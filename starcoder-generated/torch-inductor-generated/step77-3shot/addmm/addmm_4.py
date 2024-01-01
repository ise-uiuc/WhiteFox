
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp1 = torch.randn(3, 3)
    def forward(self, x1, z1, x2):
        v1 = torch.mm(x1, x2)
        v2 = v1 + self.inp1 + z1
        return v2
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
z1 = torch.randn(3, 3)
