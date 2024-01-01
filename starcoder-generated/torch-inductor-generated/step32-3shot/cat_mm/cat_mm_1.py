
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v1], 1)
        t2 = torch.cat([t1, t1], 1)
        t3 = torch.cat([t2, t2], 1)
        t4 = torch.cat([t3, t3], 1)
        t5 = torch.cat([t4, t4], 1)
        t6 = torch.cat([t5, t5], 1)
        t7 = torch.cat([t6, t6], 1)
        return torch.cat([t7, t7], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
