
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x1, x3)
        h3 = torch.mm(x2, x4)
        h4 = torch.mm(x3, x4)
        h3 = torch.mm(x1, x2)
        return h1 + h2 + h3 + h4
# Inputs to the model
x1 = torch.randn(6, 6)
x2 = torch.randn(6, 6)
x3 = torch.randn(6, 6)
x4 = torch.randn(6, 6)
