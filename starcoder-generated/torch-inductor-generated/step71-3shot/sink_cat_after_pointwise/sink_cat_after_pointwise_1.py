
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view([1, 2, 3, 4])
        y = y.view([16])
        y = y.view([-1] * y.ndim)
        return y
# Inputs to the model
x = torch.randn(5)
