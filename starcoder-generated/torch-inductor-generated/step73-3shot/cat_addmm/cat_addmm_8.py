
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 2)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(0)
        x = x.exp()
        return x
# Inputs to the model
x = torch.randn(1, 2, 3)
