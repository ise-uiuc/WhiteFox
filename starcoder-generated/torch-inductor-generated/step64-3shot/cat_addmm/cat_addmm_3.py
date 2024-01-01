
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 16),
            nn.Linear(16, 64)
        )
        self.layers2 = nn.Sequential(
            nn.Linear(2, 64),
            nn.Softmax(),
            nn.Linear(64,64)
        )
        self.layers3 = nn.Sequential(
            nn.Linear(64,64),
            nn.Softmax()
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
