
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        t1 = torch.mm(x1, x1)
        t2 = torch.mm(x1, x1)
        t3 = t1 + t2
        return t3
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
