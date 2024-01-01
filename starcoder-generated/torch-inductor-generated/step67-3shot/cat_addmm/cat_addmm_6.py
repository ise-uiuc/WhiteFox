
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.first_layers = nn.Linear(3, 4)
        self.second_layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.first_layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.flatten(start_dim=2)
        x = self.second_layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = x.flatten(start_dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 3)
