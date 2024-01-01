
class Model(torch.nn.Module):
    def __init__(self, x1, x2):
        super().__init__()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        t1 = torch.cat([v, v], 1)
        t2 = torch.cat([t1, t1], 0)
        t3 = torch.cat([t2, t2], 1)
        t4 = torch.cat([t3, t3], 1)
        t5 = torch.cat([t4, t4], 1)
        t6 = torch.cat([t5, t5], 1)
        t7 = torch.cat([t6, t6], 1)
        t8 = torch.cat([t7, t7], 1)
        t9 = torch.cat([t8, t8], 1)
        t10 = torch.cat([t9, t9], 1)
        t11 = torch.cat([t10, t10], 1)
        t12 = torch.cat([t11, t11], 1)
        t13 = torch.cat([t12, t12], 1)
        return torch.cat([t13, t13], 1)
# Inputs to the model
x1 = torch.randn(6, 5)
x2 = torch.randn(5, 6)
