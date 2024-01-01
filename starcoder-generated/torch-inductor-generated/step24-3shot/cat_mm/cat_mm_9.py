
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v1 = torch.mm(x1, x2)
        t1 = torch.cat([v1, v1], 1)
        return torch.cat([t1, t1], 1)
# Inputs to the model
x1 = torch.randn(1, 4)
x2 = torch.randn(4, 1)
