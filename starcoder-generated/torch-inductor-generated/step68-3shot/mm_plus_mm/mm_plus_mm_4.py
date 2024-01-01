
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3, x4):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x2, x1)
        h3 = torch.mm(x3, x1)
        h4 = torch.mm(x4, x1)
        return h4 + h2 + h3 + h1
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
x3 = torch.randn(5, 5)
x4 = torch.randn(5, 5)
