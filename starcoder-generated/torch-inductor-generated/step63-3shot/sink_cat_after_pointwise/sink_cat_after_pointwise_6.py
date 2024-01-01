
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x, x, x, x), dim=0)
        y = y.view(1, 5*x.shape[0], -1)
        y = torch.tanh(y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
