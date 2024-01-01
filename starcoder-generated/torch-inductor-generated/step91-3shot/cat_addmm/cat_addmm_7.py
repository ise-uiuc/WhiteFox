
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 4)
    def forward(self, x):
        x = torch.rand_like(x) * self.layers(x)
        return torch.cat([x, x, x, x], dim=1)
# Inputs to the model
x = torch.randn(2, 3)
