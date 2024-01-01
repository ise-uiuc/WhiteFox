
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5, x6):
        v1 = torch.cat([x2, x3, x4, x5, x6], 0);
        v2 = torch.cat([x3, x4, x5, x6], 0);
        v3 = torch.cat([x4, x5, x6], 0);
        v4 = torch.cat([x5, x6], 0);
        v5 = torch.cat([x6], 0);
        v6 = torch.cat([x1], 0);
        v7 = torch.mm(v1, v2)
        v8 = torch.mm(v3, v4)
        v9 = torch.mm(v5, v6)
        t1 = torch.cat([v7, v8, v9], 1)
        t2 = torch.cat([x2, x3, x4, x5], 0);
        t3 = torch.cat([t2, t2], 0);
        t4 = torch.cat([x3, x4, x5], 0);
        t5 = torch.cat([t4, t4], 0);
        t6 = torch.cat([x4, x5], 0);
        t7 = torch.cat([t6, t6], 0);
        t8 = torch.cat([x5], 0);
        t9 = torch.cat([t8], 0);
        t10 = torch.mm(t2, t3)
        t11 = torch.mm(t5, t7)
        t12 = torch.mm(t9, v6)
        t13 = torch.cat([t10, t11, t12], 1)
        return torch.cat([t13, t13, t13], 1)
# Inputs to the model
x1 = torch.randn(1, 3)
x2 = torch.randn(1, 4)
x3 = torch.randn(1, 5)
x4 = torch.randn(1, 6)
x5 = torch.randn(1, 7)
x6 = torch.randn(1, 8)
