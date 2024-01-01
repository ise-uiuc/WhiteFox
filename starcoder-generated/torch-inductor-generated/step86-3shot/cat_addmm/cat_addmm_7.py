
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x, x], dim=2)
        x = torch.cat([x, x, x, x], dim=2).squeeze()
        return x
# Inputs to the model
x = torch.randn(1, 7)
