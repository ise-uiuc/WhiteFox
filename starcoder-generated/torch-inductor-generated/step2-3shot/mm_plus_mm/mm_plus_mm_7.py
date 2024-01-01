
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x3, x4)
        v3 = v1 + v2
        return (v1, v2, v3)
# Inputs to the model
x1 = torch.randn(5, 4)
x2 = torch.randn(4, 3)
x3 = torch.randn(5, 4)
x4 = torch.randn(4, 3)
