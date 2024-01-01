
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 4)
        self.layers_2 = nn.Linear(2, 3)
    def forward(self, x1):
        x1 = self.layers(x1)
        x1 = torch.stack((x1, x1), dim=1)
        x1 = torch.stack((x1, x1), dim=1)
        x1 = torch.add(x1, self.layers_2(x1))
        return x1
# Inputs to the model
x1 = torch.randn(1, 4)
