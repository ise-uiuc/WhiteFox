
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x):
        t1 = torch.mm(x, x)
        t2 = torch.mm(x, x)
        t3 = t1 + t2
        t4 = torch.mm(x, x)
        t5 = t1 + t2 + t3 + t4
        t6 = torch.mm(x, x)
        t7 = t5 + t6
        t8 = torch.mm(x, x)
        t9 = t6 + t8
        t10 = t7 + t9
        return t10
# Inputs to the model
x = torch.randn(1, 2)
