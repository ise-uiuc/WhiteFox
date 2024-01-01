
class Model(torch.nn.Module):
    def forward(self, x1, x2, *args, x4):
        v4 = torch.mm(x4, x4)
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x2, x2)
        return v1 + v2 + v4
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
x4 = torch.randn(2, 2)
x5 = torch.randn(2, 2)
x6 = torch.randn(2, 2)
x7 = torch.randn(2, 2)
