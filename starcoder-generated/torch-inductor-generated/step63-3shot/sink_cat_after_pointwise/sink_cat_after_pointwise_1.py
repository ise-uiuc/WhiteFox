
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], int(x.shape[1] / 2), -1)
        y = y.sum(dim=2)
        return y * y
# Inputs to the model
x = torch.randn(2, 10)
