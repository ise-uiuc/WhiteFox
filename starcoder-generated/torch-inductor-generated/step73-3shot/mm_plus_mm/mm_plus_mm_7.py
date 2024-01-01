
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        out = torch.mm(x1, x3)
        out = torch.mm(x2, x3)
        out = torch.mm(x1, x2)
        return out
# Inputs to the model
x1 = torch.randn(4, 4)
x2 = torch.randn(4, 4)
x3 = torch.randn(4, 4)
