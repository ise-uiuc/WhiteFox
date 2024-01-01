
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        a1 = torch.mm(x1, x2)
        a2 = torch.mm(x3, x4)
        a2 = torch.mm(x1, x3)
        return a1 + a2
# Inputs to the model
x1 = torch.randn(3, 6)
x2 = torch.randn(3, 6)
x3 = torch.randn(6, 4)
x4 = torch.randn(6, 4)
