
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        for i in range(0, 2):
            y.tanh()
            if i == 1:
                y = torch.cat((y, y), dim=1)
        if True:
            k = y.view(x.shape[0], -1).tanh()
        else:
            k = y.view(x.shape[0], -1).tanh()
        for i in range(0, 2):
            k.sin()
            if i == 1:
                k = k.view(x.shape[0], -1).sin()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
