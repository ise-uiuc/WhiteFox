
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        y = torch.stack((x, x), dim=0)
        z = torch.stack((y, y, y), dim=0)
        q = torch.flatten(z, 1)
        q = q.reshape((1,18))
        return q
# Inputs to the model
x = torch.randn(3, 2)
