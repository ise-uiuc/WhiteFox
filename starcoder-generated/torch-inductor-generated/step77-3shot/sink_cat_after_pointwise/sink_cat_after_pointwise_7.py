
def func(x):
    y = x.reshape(x.shape)
    return y.relu()

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = func(x)
        for i in range(5):
            z = func(x)
            y = torch.cat([y, z], dim=1)
            y = y.view(y.shape[0], -1)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
