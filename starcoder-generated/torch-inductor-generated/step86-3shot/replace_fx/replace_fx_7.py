
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.rand_like(x1)
        t2 = t1 * x2
        return t2
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = 1
x3 = 1
