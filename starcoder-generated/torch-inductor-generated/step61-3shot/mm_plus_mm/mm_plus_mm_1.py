
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        y = torch.mm(x1, x2)
        v = torch.mm(y, z)
        return v
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
x4 = torch.randn(5, 5)
