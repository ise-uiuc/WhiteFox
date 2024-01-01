
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, x3, x4, x5):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x3, x4), torch.mm(x5, x2), torch.mm(x3, x2), torch.mm(x3, x1)], 1)
        v2 = torch.cat([torch.mm(x1, x2), torch.mm(x3, x4)], -1)
        return torch.cat([v1, v2, v2, v2, v2], 1)
# Inputs to the model
x1 = torch.randn(10, 20)
x2 = torch.randn(4, 20)
x3= torch.randn(3, 20)
x4= torch.randn(7, 20)
x5= torch.randn(9, 20)
