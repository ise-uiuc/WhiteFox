
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = x.expand([3, 3])
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
