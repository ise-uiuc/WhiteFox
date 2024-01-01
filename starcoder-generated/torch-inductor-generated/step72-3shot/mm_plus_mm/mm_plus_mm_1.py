
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.W1 = torch.nn.Parameter(torch.randn(20, 10))
        self.W2 = torch.nn.Parameter(torch.randn(10, 2))
    def forward(self, x1, x2):
        h1 = torch.mm(x1, self.W1)
        h2 = torch.mm(x1, self.W2)
        out = torch.mm(h1, self.W2.t()) + torch.mm(h2, self.W1.t())
        return out

# Inputs to the model
x1 = torch.randn(5, 20)
x2 = torch.randn(5, 2)

