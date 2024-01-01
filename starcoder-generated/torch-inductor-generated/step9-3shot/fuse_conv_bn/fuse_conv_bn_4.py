
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x5):
        c5 = torch.nn.Conv2d(2, 2, 1)
        v4 = c5(x5)
        return c5(v4)
# Inputs to the model
x5 = torch.randn(1, 2, 3, 3)
