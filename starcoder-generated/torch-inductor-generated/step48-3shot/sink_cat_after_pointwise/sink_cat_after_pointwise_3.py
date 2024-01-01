
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat((x, x * x), dim=1)
        if y.dim() < 2 or y.size(0) < 2:
            return y.view(y.shape[0], -1).tanh()
        else:
            return y.view(y.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(1, 4, 9)
