
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
        self.linear = nn.Linear(1, 3)
    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
