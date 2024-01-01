
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0] * x.shape[1], -1)
        y = y.tanh()
        x = y.tanh()
        return x
# Inputs to the model
x = torch.randn(3, 2, 4)
