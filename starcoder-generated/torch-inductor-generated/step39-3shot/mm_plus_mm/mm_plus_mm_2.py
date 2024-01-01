
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v2 = torch.mm(x3, x1)
        v1 = torch.mm(v2, x4)
        v2 = torch.mm(x3, x2)
        v3 = torch.mm(v2, x4)
        v4 = torch.mm(v3, x2)
        return v1 + v2 + v3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
