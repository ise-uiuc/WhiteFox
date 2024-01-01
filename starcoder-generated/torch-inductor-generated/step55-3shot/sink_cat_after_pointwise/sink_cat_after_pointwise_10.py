
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x / 10
        return x + y.view(x.shape[0], -1).tanh()
# Inputs to the model
x = torch.randn(2, 3, 4)
