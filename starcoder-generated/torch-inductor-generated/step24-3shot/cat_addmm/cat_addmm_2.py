
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.layers_2 = nn.Linear(2, 2)
    def forward(self, x):
        x = torch.stack(self.layers(x), dim=0)
        x = self.layers_2(x)
        x = torch.flatten(x, start_dim=0)
        return x
# Inputs to the model
x = torch.randn(2, 2)
