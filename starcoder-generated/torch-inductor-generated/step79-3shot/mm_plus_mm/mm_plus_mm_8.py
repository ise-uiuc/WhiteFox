
class Model(torch.nn.Module):
    def forward(self, x, y):
        h1 = torch.mm(x, x)
        h2 = torch.mm(y, y)
        h3 = torch.mm(x, x)
        return h3 + h2 + h1
# Inputs to the model
x = torch.randn(5, 5)
y = torch.randn(7, 7)
