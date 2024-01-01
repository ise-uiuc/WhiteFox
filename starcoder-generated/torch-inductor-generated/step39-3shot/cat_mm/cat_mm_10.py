
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        return torch.mm(v, x3)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
x3 = torch.randn(2, 4)
