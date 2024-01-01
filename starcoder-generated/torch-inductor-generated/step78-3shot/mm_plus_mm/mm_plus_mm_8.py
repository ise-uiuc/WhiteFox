
class Model(torch.nn.Module):
    def forward(self, a1, a2, a3, a4):
        t1 = torch.mm(a1, a2)
        t2 = torch.mm(a3, a4)
        t3 = t1 + t2
        return t3
# Inputs to the model
a1 = torch.randn(10, 5)
a2 = torch.randn(5, 5)
a3 = torch.randn(10, 5)
a4 = torch.randn(5, 5)
