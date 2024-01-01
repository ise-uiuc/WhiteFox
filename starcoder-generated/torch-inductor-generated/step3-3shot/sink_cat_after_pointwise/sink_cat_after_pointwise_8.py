
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        a = torch.zeros(*x.shape)
        y = torch.cat((x, a), dim=-1).transpose(2, 1)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[0], y.shape[1], -1).tanh()
        y = torch.cat((y, y), dim=1)
        x = y.view(y.shape[0], -1).tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
