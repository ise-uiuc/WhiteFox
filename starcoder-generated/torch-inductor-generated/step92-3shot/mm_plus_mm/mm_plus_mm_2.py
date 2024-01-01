
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.mm(x3, x4)
        t2 = torch.mm(x1, x2)
        t3 = torch.mm(x2, x1)
        return torch.mm(x1, x2) + t1
# Inputs to the model
x1 = torch.randn(10, 10)
x2 = torch.randn(10, 10)
x3 = torch.randn(10, 10)
x4 = torch.randn(10, 10)
