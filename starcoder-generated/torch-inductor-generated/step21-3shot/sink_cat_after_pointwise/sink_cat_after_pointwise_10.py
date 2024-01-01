
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = x.view(x.shape[0], -1, 32, 25)
        y = y
        x = x.view(-1, 32, 3)
        z = torch.cat((x, y), dim=1)
        return z
# Inputs to the model
x1 = torch.randn(16, 16, 32, 3)
