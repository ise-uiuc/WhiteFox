
class Model(nn.Module):
    def __init__(self): 
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_1 = nn.Linear(2, 2, True)
        self.layers_2 = nn.Linear(2, 2, False)
    def forward(self, x):
        x = self.layers(x)
        x = self.layers_1(x)
        x = self.layers_2(x)
        x = torch.cat((x, x), dim = 1)
        return x
# Model starts
x = torch.randn(1, 2)
