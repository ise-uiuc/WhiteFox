
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(16, 16)
    def forward(self, x):
        x = self.layers(x)
        x = torch.reshape(x, (-1, 4, 1, 2, 2))
        return x
# Inputs to the model
x = torch.randn(1, 16, 1, 1, 1)
