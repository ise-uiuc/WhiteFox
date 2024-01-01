
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(8, 10)
        self.linear = nn.Linear(10, 15)
    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return x
# Inputs to the model
x = torch.randn(2, 8)
