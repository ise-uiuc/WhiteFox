
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2, s1):
        v1 = torch.mm(x1, x2)
        return torch.cat([v1, v1, v1, v1, v1], s1)
# Inputs to the model
x1 = torch.randn(8, 7)
x2 = torch.randn(7, 1)
s1 = 0
