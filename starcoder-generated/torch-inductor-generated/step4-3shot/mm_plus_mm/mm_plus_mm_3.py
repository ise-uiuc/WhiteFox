
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        out = torch.mm(x1, x4) + torch.mm(x1, x4)
        out.transpose()
        return out
# Inputs to the model
x1 = torch.randn(3,2)
x2 = torch.randn(2, 5)
x3 = torch.randn(3, 2)
x4 = torch.randn(2, 5)
