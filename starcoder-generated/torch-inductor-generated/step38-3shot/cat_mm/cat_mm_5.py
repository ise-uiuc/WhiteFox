
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.cat([torch.mm(x1, x2), torch.mm(x1, x2)], 1)
        return torch.cat([
            torch.mm(v1, torch.rand(1, 1)),
            torch.mm(x1, torch.cat([x1, x2], 1)),
            torch.mm(v1, torch.rand(1, 1)),
            torch.mm(x1, torch.cat([x2, x2], 1)),
            torch.mm(v1, torch.rand(1, 1), 1),
            torch.mm(x1, torch.cat([x2, x2], 1)),
        ], 1)
# Inputs to the model
x1 = torch.randn(2, 1)
x2 = torch.randn(1, 2)
