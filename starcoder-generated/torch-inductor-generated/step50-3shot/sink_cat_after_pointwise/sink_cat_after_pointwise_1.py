
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x1, x2):
        t = torch.cat([x1, x2, x2], dim=1)
        d = t.view(t.shape[0], -1)
        c = d.clamp(0, 10, 4)
        x = x1 if c.shape == (1, 17) or c.shape == (1, 21) else x2
        return x
# Inputs to the model
x1 = torch.randn(1, 1, 1, 3, 5, 7)
x2 = torch.randn(1, 1, 2, 4, 6)
