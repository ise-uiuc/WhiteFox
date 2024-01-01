
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], x.shape[1], -1)
        y = y.permute(0, 1, 2)
        if y.dim() == 3:
            y = y.tanh()
        y = y.view(y.shape[0], -1).tanh()
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
