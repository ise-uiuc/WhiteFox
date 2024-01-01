
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        x1 = x1 + torch.mm(x1, x2)
        return x1 + torch.mm(x1, x2) + x3 + x4
# Inputs to the model
x1 = torch.randn(6, 6)
x2 = torch.randn(6, 6)
x3 = torch.randn(6, 6)
x4 = torch.randn(6, 6)
