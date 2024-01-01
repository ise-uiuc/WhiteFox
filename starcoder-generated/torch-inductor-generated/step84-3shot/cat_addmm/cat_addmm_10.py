
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.unsqueeze(-1)
        x = torch.stack((x, x, x))
        x = x.squeeze(-1).flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
