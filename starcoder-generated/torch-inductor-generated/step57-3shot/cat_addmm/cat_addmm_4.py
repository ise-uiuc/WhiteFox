
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 5)
    def forward(self, x):
        x = self.layers(x)
        x = x ** 3
        return x
# Inputs to the model
x = torch.randn(2, 2)
