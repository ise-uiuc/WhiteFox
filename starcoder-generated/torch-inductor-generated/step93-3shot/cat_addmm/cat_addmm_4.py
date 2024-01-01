
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(3, 4)
        self.layers_2 = nn.Linear(4, 4)
    def forward(self, x):
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.stack((x, x), dim=2)
        x = x.flatten(end_dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 3)
