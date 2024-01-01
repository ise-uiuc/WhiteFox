
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        y = torch.cat([x, x, x], dim=1)
        return y.view(y.shape[0], -1)
# Inputs to the model
x = torch.randn(2, 3, 4)
