
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(x.shape[0], 2, 1)
        x = torch.cat([x, x], dim=1)
        x = x.flatten(0).flatten(1)
        return x
# Inputs to the model
x = torch.randn(3, 3)
