
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 8)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x, x, x, x), dim=1)
        x = x.flatten(end_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 8)
