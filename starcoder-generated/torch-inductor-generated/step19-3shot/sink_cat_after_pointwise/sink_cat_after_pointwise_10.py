
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x
        z = x
        y = y.cat(z)
        z = y
        x = y + z.view(y.shape[0], -1)
        y = x
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
