
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x[:1], x[1:], x[1:], x[:1]], dim=3).flatten(start_dim=2).flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
