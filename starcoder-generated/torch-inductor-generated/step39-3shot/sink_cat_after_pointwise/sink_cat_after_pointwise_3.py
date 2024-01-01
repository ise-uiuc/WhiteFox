
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=-2)
        return y.view(y.shape[0], y.shape[1], -1).tanh() if y.shape!= (1, 1, 2) else y.view(y.shape[0], y.shape[1], -1).relu()
# Inputs to the model
x = torch.randn(2, 3, 4)
