
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        w = y.view(y.shape[0], -1)
        y = torch.tanh(w)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
