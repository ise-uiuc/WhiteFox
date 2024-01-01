
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(start_dim=1)
        y = torch.stack([x, x, x], dim=1)
        x = x.flatten(start_dim=1)
        return torch.cat([x, y], dim=1)
# Inputs to the model
x = torch.randn(1, 2)
