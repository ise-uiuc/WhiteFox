
class Model(torch.nn.Module):
    def forward(self, x0, x1):
        t1 = torch.mm(x0, x0)
        t2 = torch.mm(x1, t1)
        t3 = torch.mm(t2, x1)
        t4 = torch.mm(t1, t3)
        return t1 + t4
# Inputs to the model
x0 = torch.randn(16, 16)
x1 = torch.randn(16, 16)
