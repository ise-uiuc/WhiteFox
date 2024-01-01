
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        w1 = torch.cat([x1, x1], 1)
        w2 = torch.cat([x2, x2], 1)
        v1 = torch.mm(w1, w2)
        v2 = torch.mm(w1, w2)
        return torch.cat([v1, v2], 1)
# Inputs to the model
x1 = torch.randn(2, 3)
x2 = torch.randn(2, 3)
