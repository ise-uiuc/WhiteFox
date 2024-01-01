
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.tile(x, (1, 5))
        return x
# Inputs to the model
x = torch.randn(2, 2)
