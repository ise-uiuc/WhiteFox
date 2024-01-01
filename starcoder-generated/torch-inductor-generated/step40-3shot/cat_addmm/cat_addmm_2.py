
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = torch.stack((x, x), dim=1)
        x = torch.flatten(x, start_dim=2)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 4)
