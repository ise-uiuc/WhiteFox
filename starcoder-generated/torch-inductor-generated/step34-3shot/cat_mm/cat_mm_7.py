
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x2)
        t2 = torch.cat([t1, t1, t1], 1)
        t3 = torch.cat([t2, t2], 1)
        t4 = torch.cat([t3, t3], 1)
        t5 = torch.cat([t4, t4], 1)
        return torch.cat([t5, t5], 1)
# Inputs to the model
x1 = torch.randn(4, 2)
x2 = torch.randn(2, 4)
