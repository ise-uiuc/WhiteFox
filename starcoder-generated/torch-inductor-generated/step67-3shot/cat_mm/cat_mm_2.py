
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        v2 = torch.mm(x1, x2)
        v3 = torch.mm(x1, x2)
        return torch.cat([v3, v2, v1], 0)
# Inputs to the model
x1 = torch.randint(1, (5, 2, 3))
x2 = torch.randint(1, (1, 3))
