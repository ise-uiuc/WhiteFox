
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers_1 = nn.Linear(32, 75)
    def forward(self, x):
        x = self.layers_1(x)
        x = x.flatten(start_dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 32, 1, 1)
