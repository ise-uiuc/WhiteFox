
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = y.view(x.shape[0], -1)
        if x.shape!= (1, 3): y = y.tanh()
        return y.tanh() if y.shape!= (1, 3) else y.relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
