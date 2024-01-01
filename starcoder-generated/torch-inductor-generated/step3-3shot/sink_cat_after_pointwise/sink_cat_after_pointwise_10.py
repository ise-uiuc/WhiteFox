
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        if y.dim() == 2:
            y1 = y.tanh()
            y2 = y1.tanh()
            y = torch.cat((y1, y2), dim=0)
        else:
            y = y.view(x.shape[0], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
