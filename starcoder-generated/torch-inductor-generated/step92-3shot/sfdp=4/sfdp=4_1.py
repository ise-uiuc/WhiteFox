
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X_r, a, B, c):
        x_r = X_r * a
        b = b * (c > 0).type(c.dtype)
        d = x_r * b
        output = d.sum()
        return output
# Inputs to the model
X_r5 = torch.randn(32, 1, 3, 3)
a6 = torch.randn(32, 1, 3, 3)
b4 = torch.randn(32, 1, 3, 3)
c1 = torch.randn(32, 1, 3, 3, 3)
