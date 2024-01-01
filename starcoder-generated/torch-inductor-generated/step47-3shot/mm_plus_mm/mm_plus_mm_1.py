
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t1 = torch.mm(x1, x1)
        t2 = torch.mm(x2, x3)
        t3 = t1 + t2
        t4 = torch.mm(x1, x2)
        t5 = torch.mm(x2, x1)
        t6 = t4 + t5
        t7 = t2 + t6
        return t7 + t3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
x3 = torch.randn(3, 3)
