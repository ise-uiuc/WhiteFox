
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, X, Y):  # X: shape [1, 300, 0]   Y: shape [32, 0, 400]
        # TODO: add computation here
        return torch.stack([torch.cat([torch.mm(x, y) for x in X], dim=0) for y in Y], dim=1)
# Inputs to the model
X = torch.randn(1, 300, 50)
Y = torch.randn(32, 50, 400)
