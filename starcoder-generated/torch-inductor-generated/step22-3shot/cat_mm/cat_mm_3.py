
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        m1 = list()
        m2 = list()
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        for i in range(7):
            t1 = torch.cat([v1, v1, v1, v1, v1, v1, v1], 1)
            t2 = torch.cat([v2, v2, v2], 1)
            m1.append(t1)
            m2.append(t2)
        t3 = torch.cat(m1, 1)
        t4 = torch.cat(m2, 1)
        t5 = torch.cat([t3, t4], 1)
        return t5
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(3, 1)
