
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1)
        x = y.view(x.shape[0], -1).tanh() + y.view(x.shape[0], -1).sigmoid().add(x.view(x.shape[0], -1)).abs().add(2).log()
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
