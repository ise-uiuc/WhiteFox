
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 1)
        self.layers.bias = None
    def forward(self, x):
        x = self.layers(x)
        x = x.view(1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
