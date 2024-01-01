
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0), torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0), torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 0)], 1)
        v2 = torch.cat([torch.mm(x1, x2), v1, torch.mm(x1, x2)], 0)
        v3 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2), v2], 1)
        v4 = torch.cat([v3, v3], 0)
        return v4
# Inputs to the model
x1 = torch.randn(2, 4)
x2 = torch.randn(4, 1)
