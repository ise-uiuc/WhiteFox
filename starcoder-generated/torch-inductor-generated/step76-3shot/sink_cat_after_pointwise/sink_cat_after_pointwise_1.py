
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x, x], dim=1)
        y = y.view(3, -1)
        return y.relu()
# Inputs to the model
x = torch.randn(4, 3, 4)
