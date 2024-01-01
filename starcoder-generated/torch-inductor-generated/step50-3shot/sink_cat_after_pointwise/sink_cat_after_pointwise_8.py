
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x, x, x, x, x, x, x, x], dim=0)
        x = y.view(y.shape[1], -1)
        return x.tanh() if y.shape!= (10, 12) else x.relu()
# Inputs to the model
x = torch.randn(4, 3, 4)
