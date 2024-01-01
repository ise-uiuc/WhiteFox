
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        v = []
        for i in range(0, 3):
            v += [torch.cat([x1, x2], 1)]
        return torch.cat(v, 0)
# Inputs to the model
x1 = torch.randn(1, 2)
x2 = torch.randn(1, 2)
