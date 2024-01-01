
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        t0 = torch.mm(x1, x2)
        t1 = torch.mm(x1, x3)
        t2 = torch.mm(x3, x1)
        out = t0 + t1 + t2
        return out + out + out
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
