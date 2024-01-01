
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1).tanh()
        x = x.tanh().view(x.shape[0], -1).tanh()
        return y.cat([x, x], dim=1)
# Inputs to the model
x = torch.randn(1, 2, 3, 2)
