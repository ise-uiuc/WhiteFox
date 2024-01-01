
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layer = nn.Linear(2, 3)
        self.second_layer = nn.Conv2d(2, 6, 2, padding = (0, 1))
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.first_layer(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.flatten(start_dim=1)
        x = self.second_layer(x)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 1)
