
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim =0)
        y = x.view(y.shape[0], -1)
        x = y.tanh()
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
