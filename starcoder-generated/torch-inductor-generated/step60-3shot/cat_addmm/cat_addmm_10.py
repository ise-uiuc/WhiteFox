
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.flatten(start_dim=1)
        x = x + x
        x = x.chunk(2, dim=1)
        x = torch.cat(x, dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
