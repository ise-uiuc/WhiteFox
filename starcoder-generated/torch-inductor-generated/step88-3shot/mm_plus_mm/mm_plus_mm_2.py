
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        t1 = torch.mm(x1, x2)
        t2 = torch.mm(x1, x2)
        return t1 + t2
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
