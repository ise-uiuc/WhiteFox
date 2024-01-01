
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        v1 = torch.mm(x1, x2) + torch.mm(x3, x4)
        v2 = torch.mm(x3, x4) + torch.mm(x1, x2)
        v3 = v1 * v2
        v4 = torch.mm(x2, x1) * torch.mm(x4, x3)
        return v3 + v4
# Inputs to the model
x1 = torch.randn(65, 65)
x2 = torch.randn(65, 65)
x3 = torch.randn(3, 3)
x4 = torch.randn(3, 3)
