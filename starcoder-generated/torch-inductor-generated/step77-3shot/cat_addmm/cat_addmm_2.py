
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers_2(x)
        x, x = torch.split(x, 1, dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
