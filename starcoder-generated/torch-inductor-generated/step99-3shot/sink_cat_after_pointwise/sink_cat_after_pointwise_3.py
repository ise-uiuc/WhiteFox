
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x, x], dim=0)
        y = y.view(y.shape[0], 2, int(y.shape[2]/2))
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
