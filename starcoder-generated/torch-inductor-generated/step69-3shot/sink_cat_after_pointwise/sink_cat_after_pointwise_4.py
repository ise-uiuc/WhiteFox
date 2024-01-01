
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        y = torch.cat([x, x], dim=1)
        z = [y, y, y]
        x = torch.stack(z).view(3, x.shape[0], -1)
        del z
        x = torch.sqrt(x) if x.shape!= (3, 2, 12) else torch.tanh(x)
        del y
        return x
# Inputs to the model
x = torch.randn(2, 3, 4)
