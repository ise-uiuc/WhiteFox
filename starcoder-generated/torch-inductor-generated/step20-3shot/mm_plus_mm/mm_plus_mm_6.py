
class Model(nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x1)
        v2 = x2.mm(x2)
        v3 = torch.mm(x3, x2)
        v4 = x4.mm(x3)
        return v1 * v2 + v3 + v4
# Inputs to the model
x1 = torch.randn(100, 55)
x2 = torch.randn(55, 100)
x3 = torch.randn(100, 55)
x4 = torch.randn(55, 100)
