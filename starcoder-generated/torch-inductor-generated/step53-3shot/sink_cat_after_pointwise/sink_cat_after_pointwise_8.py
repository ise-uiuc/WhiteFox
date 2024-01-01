
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=3)
        return y.view(y.shape[0], y.shape[1], -1).tanh() if y.shape!= (2, 3, 2, 1) else y.view(y.shape[0], y.shape[1], -1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
