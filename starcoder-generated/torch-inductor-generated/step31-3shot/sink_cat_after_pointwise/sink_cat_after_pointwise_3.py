
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = 2*x
        y = y.view(x.shape[0], -1)
        y = y.tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
