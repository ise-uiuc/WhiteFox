
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        x = x.relu()
        y = y.tanh() if y.shape == (2, 8) else y.relu()
        y = y.view(2, -1)
        y = y.sigmoid()
        z = x + y
        z = z.view(z.shape[0], -1)
        ret = z.tanh() + 3 * z.sigmoid() - 6 * z.relu()
        return ret
# Inputs to the model
x = torch.randn(1, 2, 16, 16)
y = torch.randn(1, 2, 16, 16)
