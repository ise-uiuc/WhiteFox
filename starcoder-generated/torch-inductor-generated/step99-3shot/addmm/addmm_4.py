
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.x1 = torch.randn(3, 3)
    def forward(self, x2, inp):
        v1 = torch.mm(x2, self.x1)
        v2 = v1 + self.x1 + inp
        return v2
# Inputs to the model
x2 = torch.randn(3, 3)
inp = torch.randn(3, 3)
