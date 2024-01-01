
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.mm(x1, x3)
        v2 = torch.mm(x4, x5)
        v3 = torch.mm(x1, x5)
        v4 = torch.mm(x3, x4)
        return (v1 + v2 + v3 + v4)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
x4 = torch.randn(1, 1)
x5 = torch.randn(1, 1)
