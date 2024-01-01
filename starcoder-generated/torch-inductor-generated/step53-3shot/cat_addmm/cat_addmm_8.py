
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(10, 20)
        self.linear = nn.Linear(10, 20, bias = False)
    def forward(self, x):
        x = self.layers(x)
        x = self.linear(x)
        return x
# Inputs to the model
x = torch.randn(6, 10)
