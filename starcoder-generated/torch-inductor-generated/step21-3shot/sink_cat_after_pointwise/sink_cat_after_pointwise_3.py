
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y1 = x.view(x.shape[0], -1)
        y2 = y1.tanh()
        if y1.dim() == 2:
            y1 = y1.tanh()
        else:
            y1 = x.view(x.shape[0], -1).tanh()
        y = torch.cat((y1, y2), dim=-1)
        x = y.tanh()
        return x
# Inputs to the model
x = torch.randn(6, 3, 4)
