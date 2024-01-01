
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(10)
    def forward(self, x1, x2):
        z1 = torch.mm(x1, x2) + self.inp + x2 + x1
        return x1 + x2 + z1
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
