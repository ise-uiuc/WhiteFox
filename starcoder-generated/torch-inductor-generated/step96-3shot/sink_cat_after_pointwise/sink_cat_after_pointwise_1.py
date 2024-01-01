
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x1 = torch.cat([x, x, x], dim=1)
        x2 = x1.view(x1.shape[0], -1)
        x3 = x2.tanh()
        x4 = x3.tanh()
        x5 = x4.tanh()
        x6 = x5.tanh()
        return x6
# Inputs to the model
x = torch.randn(2, 1, 4)
