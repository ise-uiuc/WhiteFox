
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(3)
    def forward(self, x1, x2):
        v1 = x1 * torch.mm(self.inp, x1)
        return x2 + v1
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3)
