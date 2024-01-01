
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        v1 = torch.mm(x1, x2)
        for loopVar6 in range(5):
            v.append(torch.cat([v1, v1, v1, v1, v1, v1, v1, v1], 1))
        return torch.cat(v, 1)
# Inputs to the model
x1 = torch.randn(1, 1)
x2 = torch.randn(1, 1)
