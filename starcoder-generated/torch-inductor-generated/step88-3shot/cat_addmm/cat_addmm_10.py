
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
    def forward(self, x):
        x = self.layers(x).flatten(2)
        return x
# Inputs to the model
x = torch.randn(2, 3)
