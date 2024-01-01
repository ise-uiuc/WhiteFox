
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(4, 3)
        self.layers_2 = nn.Linear(3, 1)
    def forward(self, x):
        x = self.layers_1(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.stack([x, x, x], dim=1)
        x = torch.flatten(x, start_dim=1)
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 4)
