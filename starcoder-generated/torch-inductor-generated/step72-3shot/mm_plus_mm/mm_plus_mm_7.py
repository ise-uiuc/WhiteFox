
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
    def forward(self, x1, x2):
        h1 = torch.mm(x1, x2)
        h2 = torch.mm(x2, x1)
        out = torch.mm(x1, x2)
        return h1 + h2 + out
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
