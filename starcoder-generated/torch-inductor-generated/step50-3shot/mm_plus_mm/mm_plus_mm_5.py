
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x3, x2)
        v2 = torch.mm(x4, x3)
        return v1 + v2
# Inputs to the model
x1 = torch.randn(12, 12)
x2 = torch.randn(12, 12)
x3 = torch.randn(12, 12)
x4 = torch.randn(12, 12)
