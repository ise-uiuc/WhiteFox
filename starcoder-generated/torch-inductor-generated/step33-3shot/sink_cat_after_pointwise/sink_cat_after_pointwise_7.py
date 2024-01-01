
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.permute((0, 2, 3, 1))
        y = y.view(y.shape[0], -1)
        return y
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
