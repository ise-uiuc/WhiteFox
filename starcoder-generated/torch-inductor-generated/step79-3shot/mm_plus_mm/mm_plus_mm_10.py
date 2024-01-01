
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        x2 = torch.mm(x2, x3)
        x1 = torch.mm(x1, x4)
        x4 = torch.mm(x3, x2)
        return x1 + x2 + x3 + x4
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
x4 = torch.randn(4, 4)
