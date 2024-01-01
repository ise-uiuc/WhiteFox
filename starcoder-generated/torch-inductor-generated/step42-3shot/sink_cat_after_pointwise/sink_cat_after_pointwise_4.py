
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x + x
        x = x + x
        y = torch.cat([x, x], dim=1)
        y = y.view(6).tanh() if (y.shape)!= (6,) else (y.view(-2).tanh())
        y = y * y
        return y
# Inputs to the model
x = torch.randn(2, 2, 2)
