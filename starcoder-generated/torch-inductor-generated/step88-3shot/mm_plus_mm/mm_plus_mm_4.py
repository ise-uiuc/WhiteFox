
class Model(torch.nn.Module):
    def forward(self, x1):
        h1 = torch.mm(x1, x1)
        h2 = torch.mm(x1, x1)
        for i in [0, 1]:
            h1 = torch.mm(h1, h1)
        return h1 + h2
# Inputs to the model
x1 = torch.randn(6, 6)
