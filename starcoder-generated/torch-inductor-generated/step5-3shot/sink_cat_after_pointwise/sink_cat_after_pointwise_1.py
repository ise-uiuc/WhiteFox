
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        y = y.tanh()
        y = torch.cat((y, y), dim=1)
        if len(x.size) == 3:
            y = y.tanh()
        else:
            y = y.view(-1).tanh()
        x = y.view(-1)
        if len(x.size) == 3:
            x = y.tanh()
        else:
            x = x.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
