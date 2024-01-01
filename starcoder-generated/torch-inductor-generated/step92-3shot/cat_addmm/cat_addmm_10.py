
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(2, 1)
        self.layers_2 = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.cat([x, x], dim=0)
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
