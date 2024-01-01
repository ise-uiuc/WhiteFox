
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
        self.layer = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 1)
        )
    def forward(self, x):
        x = self.layers(x)
        x = self.layer(x)
        return x
# Inputs to the model
x = torch.randn(1, 2)
