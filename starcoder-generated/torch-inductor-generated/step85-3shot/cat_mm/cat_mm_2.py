
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = list()
    def forward(self, x1, x2):
        v = torch.mm(x1, x2)
        v = torch.mm(x1, x2)
        v = torch.mm(x1, x2)
        return torch.cat([v, v, v], 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
