
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y, z):
        y = torch.cat((x, y, z), dim=1).view(y.shape[0], -1)
        x = y.tanh() if y.shape == (1, 2) else y.tanh()
        return x
# Inputs to the model
x = torch.randn(3, 2)
y = torch.randn(3, 2)
z = torch.randn(2, 2)
