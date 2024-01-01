
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(1, 3)
        self.layers_2 = nn.Linear(3, 1)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.stack((x, x, x), dim=1).flatten(1)
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 1)
