
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, z):
        y = x.view(x.size(1), -1)
        y = z * z
        y = y.sum().mean()
        return y
# Inputs to the model
x = torch.randn(2, 10, 20)
z = x + 1
