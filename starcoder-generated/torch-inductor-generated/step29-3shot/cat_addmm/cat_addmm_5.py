
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x[:, 0], x[:, 1]], dim=1)
        x = torch.stack([x[:, 0], x[:, 1]], dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 2)
