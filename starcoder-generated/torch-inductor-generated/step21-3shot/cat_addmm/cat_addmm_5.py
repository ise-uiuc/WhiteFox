
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.max(x, dim=2)
        return x
# Inputs to the model
x = torch.randn(4, 5, 6)
