
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        # y.dim() - 2 is needed
        y = x.view(x.shape[0], -1)
        if y.dim() - 2 > 2:
            y = y.tanh()
        else:
            y = y.view(x.shape[0], -1).tanh()
        x = torch.cat((x, y), dim=1)
        y = x.view(x.shape[0],-1).tanh()
        y = torch.cat((y, y), dim=-2)
        x = x.view(-1)
        return y
# Inputs to the model
x = torch.randn(1, 3, 2)
