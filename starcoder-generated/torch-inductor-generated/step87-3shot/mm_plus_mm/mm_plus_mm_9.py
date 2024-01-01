
class Model(torch.nn.Module):
    def forward(self, x1, x2):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x1, x2)
        h3 = h1 + h2
        return h3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
