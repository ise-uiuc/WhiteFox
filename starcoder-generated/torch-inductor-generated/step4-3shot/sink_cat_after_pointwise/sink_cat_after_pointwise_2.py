
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat((x, x), dim=0)
        y = x.transpose(0, 2)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[1], y.shape[0], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
