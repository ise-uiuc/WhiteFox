
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        t3 = torch.mm(x1, x2)
        t4 = torch.mm(x2, x1)
        t5 = t1 + t2
        t6 = t3 + t4
        return t5 + torch.mm(x1, x2) + t6
# Inputs to the model
x1 = torch.randn(20, 20)
x2 = torch.randn(20, 20)
x3 = torch.randn(20, 20)
x4 = torch.randn(20, 20)
