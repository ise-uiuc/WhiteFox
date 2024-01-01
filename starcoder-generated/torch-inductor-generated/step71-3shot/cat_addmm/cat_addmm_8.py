
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Flatten()
    def forward(self, x):
        x = self.layers(x)
        x = torch.div(x, 3.14)
        x = torch.add(1.9, x)
        x = torch.tanh(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
