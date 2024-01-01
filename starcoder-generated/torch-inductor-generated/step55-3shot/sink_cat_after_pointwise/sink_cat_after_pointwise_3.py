
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = 2*x + y.tanh()
        x = y + y
        x = torch.cat((x, x), dim=1)
        return x.view(x.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
