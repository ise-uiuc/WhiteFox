
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.split(x, 2, dim=1)
        x = x[0]
        return x
# Inputs to the model
x = torch.randn(2, 2)
