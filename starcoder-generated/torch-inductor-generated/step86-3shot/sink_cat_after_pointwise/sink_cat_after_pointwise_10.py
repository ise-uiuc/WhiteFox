
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x), dim=1)
        y = y.tanh()
        y = y.view(y.shape[0], -1)
        y = y.tanh()
        y = y.permute((1, 0))
        y = y.view(y.shape[0], -1)
        y = y.permute((1, 0))
        y = torch.cat((y, y), dim=1).tanh()
        return y
# Inputs to the model
x = torch.randn(4, 3, 4)
