
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.add(x, x, alpha=0.5)
        return x
# Inputs to the model
x = torch.randn(1, 2)
