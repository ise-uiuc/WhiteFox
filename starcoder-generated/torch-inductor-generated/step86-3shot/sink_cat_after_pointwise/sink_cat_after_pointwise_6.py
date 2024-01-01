
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        y = x.tanh()
        x = x.view(x.shape[0], -1)
        y = y.tanh()
        y = torch.cat((x, y), dim=1)
        y = y.tanh()
        y = y.view(x.shape[0], -1)
        y = y.tanh()
        z = y.relu()
        return z
# Inputs to the model
x = torch.randn(2, 3, 4)
