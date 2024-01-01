
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        # t2 = t1 + x
        t3 = torch.mm(t1, x)
        t4 = torch.mm(t1, t3)
        t5 = t4 + t4
        t6 = t5 * 0.1
        t7 = torch.mm(t5, t6)
        t8 = torch.mm(t7, t6)
        t9 = t8 * t4
        t10 = t9 + t7
        t11 = t10 * t7
        t12 = t10 * t6
        t13 = t12 + t12
        ret = t11 * t13
        return ret
# Inputs to the model
x = torch.randn(2, 2)
