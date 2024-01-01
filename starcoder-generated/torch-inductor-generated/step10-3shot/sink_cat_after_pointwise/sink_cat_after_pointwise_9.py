
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if True:
            k = y.unsqueeze(-1).expand(-1, -1, 3)
        else:
            k = y.unsqueeze(-1).expand(-1, -1, 2)
        for i in range(0, 2):
            if i == 1:
                y = torch.cat((y, y), dim=1)
        k = y.view(x.shape[0], -1).tanh()
        for i in range(0, 2):
            k.sin()
            if i == 1:
                k = k.view(x.shape[0], -1).sin()
        y = k.view(y.shape[0], -1).tanh()
        y = y.sum(dim=1, keepdim=True)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
