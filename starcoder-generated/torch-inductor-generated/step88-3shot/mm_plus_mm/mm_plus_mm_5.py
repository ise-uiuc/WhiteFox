
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4, x5, x6):
        f1 = torch.add(x1, x2)
        f2 = torch.sub(x3, x4)
        return torch.mul(f1, f2)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
x3 = torch.randn(1, 1)
x4 = torch.randn(1, 1)
x5 = torch.randn(1, 1)
x6 = torch.randn(1, 1)
