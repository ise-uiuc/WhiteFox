
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x3)
        v2 = torch.mm(x2, x4)
        v3 = v1 + v2
        return v3
# Inputs to the model
x1 = torch.randn(5, 3)
x2 = torch.randn(3, 5)
x3 = torch.randn(5, 3)
x4 = torch.randn(3, 5)
