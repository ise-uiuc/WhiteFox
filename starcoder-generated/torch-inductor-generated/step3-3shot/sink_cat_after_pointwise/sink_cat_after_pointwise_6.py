
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if y.dim() == 2:
            y = y.tanh()
        else:
            y = y.view(x.shape[0], -1).tanh()
        x = torch.cat((y, y), dim=1)
        x = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(1, 8)
