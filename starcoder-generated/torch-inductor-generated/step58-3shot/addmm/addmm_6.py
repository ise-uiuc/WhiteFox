
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inp = torch.randn(4)
    def forward(self, x1, x2):
        t1 = torch.mul(self.inp, self.inp)
        v1 = t1 + x1
        return v1
# Inputs to the model
x1 = torch.randn(3)
x2 = torch.randn(3)
