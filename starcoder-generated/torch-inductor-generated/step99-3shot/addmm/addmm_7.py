
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3, 3)
    def forward(self, z1, z2, y):
        v1 = torch.mm(z1, z2)
        v1 = torch.mm(self.inp, v1)
        v2 = torch.mm(y, self.inp)
        return v1 + v2
# Inputs to the model
z1 = torch.randn(3, 3)
z2 = torch.randn(3, 3)
y = torch.randn(3, 3)
