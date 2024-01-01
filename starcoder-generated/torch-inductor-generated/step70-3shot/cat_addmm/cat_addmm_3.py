
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 4)
        self.fc = nn.Linear(4, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x, x, x, x], dim=1)
        x = torch.cat([x, x, x, x], dim=2)
        x = torch.cat([x, x, x, x], dim=1)
        x = self.fc(x)
        return x
# Inputs to the model
x = torch.randn(1, 1)
