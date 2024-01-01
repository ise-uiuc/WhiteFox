
class Model(torch.nn.Module):
    def forward(self, x1, x2, x3):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x3, x1)
        return h1 + h2
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 2)
