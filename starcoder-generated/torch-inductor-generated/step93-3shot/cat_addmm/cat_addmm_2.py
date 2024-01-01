
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 100)
        self.layers_2 = nn.Linear(100, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = x.view(2, -1)
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
