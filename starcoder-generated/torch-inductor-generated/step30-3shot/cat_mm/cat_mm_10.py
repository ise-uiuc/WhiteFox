
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        x1.add_(12)
        x2.add_(13)
        v1 = torch.mm(x1, x2)
        x2.add_(13)
        x1.sub_(42)
        v2 = torch.mm(x1, x2)
        x2.sub_(13)
        x1.sub_(42)
        x2.add_(13)
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(3, 3)
x2 = torch.randn(3, 3)
