
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        a1 = x1.permute(0, 2, 1)[0]
        a2 = x2.permute(0, 2, 1)[0]
        b = a1 + a2
        c = b.unsqueeze(0)
        d = c.unsqueeze(0)
        e0 = d[0][0]
        return e0
# Inputs to the model
x1 = torch.randn(1, 2, 2)
x2 = torch.randn(1, 2, 2)
