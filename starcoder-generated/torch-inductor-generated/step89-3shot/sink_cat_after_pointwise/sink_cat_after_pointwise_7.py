
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        y = y.tanh() if y.shape == (2, 8) or y.shape == (2, 16) else y.relu()
        y = y.view(y.shape[0], -1)
        ret = torch.relu(y)
        return ret
# Inputs to the model
x = torch.randn(32, 32, 3)
