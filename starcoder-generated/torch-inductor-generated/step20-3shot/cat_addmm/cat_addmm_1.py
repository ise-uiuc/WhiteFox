
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = self.layers_2(x)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
