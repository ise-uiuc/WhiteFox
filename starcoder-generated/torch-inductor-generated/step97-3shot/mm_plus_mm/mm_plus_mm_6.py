
class Model(torch.nn.Module):
    def forward(x1, x2, x3, x4):
        t1 = x1.mm(x2)
        t2 = torch.mm(x1,x3)
        t3 = x4.mm(x1)
        t4 = x4.mm(x2)
        return t1 + t2 + t3 + t4
# Inputs to the model
x1 = torch.randn(50, 50)
x2 = torch.randn(50, 50)
x3 = torch.randn(50, 50)
x4 = torch.randn(50, 50)
