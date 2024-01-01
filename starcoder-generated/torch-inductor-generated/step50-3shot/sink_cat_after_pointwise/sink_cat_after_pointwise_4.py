
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.tanh() if y.shape == (2, 8) or y.shape == (1, 6) else y.relu()
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
