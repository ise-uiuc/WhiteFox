
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.arange(3)
        y = y.repeat(x.shape[0],)
        z = torch.cat((x, x), dim=1)
        y = z.view(z.shape[0], -1)
        y = y.tanh()
        x = y.relu()
        return x
# Inputs to the model
x = torch.randn(3, 2, 2)
