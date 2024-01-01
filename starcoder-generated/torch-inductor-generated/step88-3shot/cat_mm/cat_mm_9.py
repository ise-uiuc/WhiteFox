
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1):
        v = []
        idx = 0
        while len(v) < 10:
          v.append(idx)
          idx += 1
        return torch.cat([v], 0)
# Inputs to the model
x1 = torch.randn(4, 4)
