
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.tanh()
        x = torch.cat((y, y), dim=1).tanh() if y.shape[0] == 1 else torch.cat((y, y), dim=1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
