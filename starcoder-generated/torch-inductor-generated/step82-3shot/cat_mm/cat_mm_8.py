
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mul(t1, t1)
        t3 = torch.mm(x, x)
        return torch.cat([t1, t2, t3], 1)
# Inputs to the model
x = torch.randn(2, 4)
