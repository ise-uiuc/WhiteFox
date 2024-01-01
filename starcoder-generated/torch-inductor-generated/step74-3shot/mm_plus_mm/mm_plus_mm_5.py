
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        t3 = torch.mm(x3, x1)
        t4 = torch.mm(x2, x3)
        return t1 + t2 + t3 + t4
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
