
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=-2)
        z = y.view(y.shape[0], y.shape[1], -1).tanh()
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
