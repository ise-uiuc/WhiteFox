
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(24, 96)
        self.layers_2 = nn.Linear(96, 48)
        self.layers_3 = nn.Linear(48, 20)
        self.layers_4 = nn.Linear(20, 9)
        self.layers_5 = nn.Linear(9, 3)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers_2(x)
        x = self.layers_3(x)
        x = self.layers_4(x)
        x = self.layers_5(x)
        return x
# Inputs to the model
x = torch.randn(2, 24)
