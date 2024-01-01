
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        for i in range(2):
            if i > 100:
                v = torch.mm(x1, x2)
        return torch.cat([v, v, v, v, v], 1)
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
