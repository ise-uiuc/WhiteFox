
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Tanh(),
            nn.Tanh()
        )
    def forward(self, x):
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
