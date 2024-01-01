
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[1], -1, x.shape[0])
        y = torch.clamp(y, 0, 5)
        return y.tanh()
# Inputs to the model
x = torch.randn(2, 2, 2)
