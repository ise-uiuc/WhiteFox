
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v2 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        v3 = torch.cat([torch.mm(v1, v2), torch.mm(v1, v2), torch.mm(v1, v2), torch.mm(v1, v2), torch.mm(v1, v2), torch.mm(v1, v2), torch.mm(torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)]), torch.mm(x1, x2)), torch.mm(x1, x2), torch.mm(x1, x2), torch.mm(x1, x2)
                       ], 1)
        v4 = torch.cat([torch.mm(torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)]), torch.mm(x1, x2)), torch.mm(x1, x2)], 1)
        return torch.cat([torch.mm(x1, x2), torch.mm(x1, x2), torch.mm(torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)]), torch.mm(x1, x2)), v4, torch.mm(x1, x2)], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
