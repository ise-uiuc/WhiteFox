
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.t3 = torch.ones((1, 2))
    def forward(self, x):
        x = self.layers(x)
        y = self.layers(self.t3)
        x = torch.add(x, y)
        x = x.flatten(start_dim=1)
        x = torch.stack([x, x], dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
