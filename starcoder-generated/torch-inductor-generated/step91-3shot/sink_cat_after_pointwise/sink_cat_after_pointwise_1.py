
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        z = y.view(y.shape[0], -1)
        if y.shape == (2, 6):
            return y.relu()
        elif y.shape!= (2, 6):
            return z.tanh()
        return z.relu()
# Inputs to the model
x = torch.randn(2, 2)
