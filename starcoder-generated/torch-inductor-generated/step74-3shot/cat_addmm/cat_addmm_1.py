
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
        self.activation_function = nn.ReLU()
    def forward(self, x):
        x = self.layers(x)
        x = self.activation_function(x)
        x = torch.cat([x, x], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
