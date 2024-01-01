
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mm(x, x)
        t3 = torch.mm(x, x)
        v1 = torch.cat([t1, t2, t3, t1, t2, t1], 1)
        return v1
# Inputs to the model
x = torch.randn(6, 5)
