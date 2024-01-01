
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v5 = torch.mm(x1, x2)
        v6 = torch.mm(x2, x3)
        return v5 + v6
# Inputs to the model
x1 = torch.randn(8, 8)
x2 = torch.randn(8, 8)
x3 = torch.randn(8, 8)
x4 = torch.randn(8, 8)
