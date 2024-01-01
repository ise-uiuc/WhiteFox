
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        x = y.view(y.shape)
        if y.shape == (1, 12):
            x = x.tanh()
        else:
            x = x[0]
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
