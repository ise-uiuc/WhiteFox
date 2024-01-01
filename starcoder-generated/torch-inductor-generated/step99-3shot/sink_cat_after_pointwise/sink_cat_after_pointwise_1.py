
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        x = torch.abs(torch.max(x, x))
        y = torch.cat((x, x, x, x), dim=1)
        y = 2 * y
        y = torch.min(x, y)
        return y
# Inputs to the model
x = torch.randn(2, 3, 4)
