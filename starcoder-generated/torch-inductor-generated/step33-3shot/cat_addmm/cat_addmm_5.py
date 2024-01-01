
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.cat([x, x], dim=-1)
        return x
# Inputs to the model
x = torch.randn(2, 2)

x = torch.tensor([[1], [2]])
model.layers.weight = torch.nn.Parameter(torch.ones(6, 1))
model.forward(x)

x = torch.tensor([[1], [2]])
model.layers.weight = torch.nn.Parameter(torch.ones(3, 3))
model.forward(x)
