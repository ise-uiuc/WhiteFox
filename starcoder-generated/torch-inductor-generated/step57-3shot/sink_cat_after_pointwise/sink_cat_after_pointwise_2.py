
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.cat([x, x, x], dim=None)
        x = x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], 3, 2, 4).tanh()
# Inputs to the model
x = torch.randn(1, 3, 4)
