
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t0 = torch.mm(x1, x2)
        t1 = torch.cat([t0, t0, t0], 1)
        return t1
# Inputs to the model
x1 = torch.randn(4, 3)
x2 = torch.randn(3, 3)
