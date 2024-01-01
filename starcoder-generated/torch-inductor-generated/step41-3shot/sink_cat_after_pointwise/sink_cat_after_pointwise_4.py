
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, y):
        z = torch.cat((x, y), dim=0)
        w = z + 2 * y
        return w
# Inputs to the model
x = torch.randn(3, 1)
y = torch.ones(3, 1)
