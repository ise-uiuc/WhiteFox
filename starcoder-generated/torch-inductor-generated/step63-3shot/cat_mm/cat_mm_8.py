
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x1)
        t2 = x2 + x1
        x1.add_(11)
        x2.sub_(17)
        return torch.cat([v1, v1], 1)
# Inputs to the model
x1 = torch.randn(5, 5)
x2 = torch.randn(5, 5)
