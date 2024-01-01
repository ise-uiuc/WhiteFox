
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x3, x2.t())], 0)
        v2 = torch.cat([torch.mm(x2, x2), torch.mm(x3, x1.t())], 0)
        return torch.cat([v1, v2, v2], 0)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(4, 4)
x3 = torch.randn(2, 3)
