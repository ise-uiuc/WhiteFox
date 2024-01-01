
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = y.tanh() if y.shape == (2, 8) else y.relu()
        return y.view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(32, 32, 3)
