
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.layers_2 = nn.Linear(3, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.transpose(0, 1)
        x = x.flatten(start_dim=2, end_dim=3)
        x = torch.stack([x, x, x], dim=1)
        x = torch.abs(x)
        x = self.layers_2(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
