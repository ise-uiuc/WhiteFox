
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t = torch.mm(x1, x2)
        t = torch.mm(t, x2)
        v1 = torch.cat([t, t, t, t, t, t], 0)
        v2 = torch.cat([t, t, t, t, t, t], 0)
        v3 = torch.cat([t, t, t, t, t, t], 0)
        return v1, v2, v3
# Inputs to the model
x1 = torch.randn(2, 2)
x2 = torch.randn(2, 2)
