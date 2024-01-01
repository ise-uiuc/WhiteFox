
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(4, 3)
        self.layers_2 = nn.Linear(3, 3)
        self.layers_3 = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.stack((x, x), dim=1)
        x = self.layers_2(x)
        x = torch.stack((x, x, x, x), dim=1)
        y = self.layers_3(x)
        y = y.unsqueeze(1)
        y = y.flatten(2)
        return x
# Inputs to the model
x = torch.randn(2, 4)
