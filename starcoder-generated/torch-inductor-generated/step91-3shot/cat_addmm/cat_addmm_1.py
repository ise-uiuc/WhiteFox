
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 8)
        self.stack = torch.stack
    def forward(self, x):
        x = torch.cat((x, x, x, x), dim=1)
        x = self.layers(x)
        x = self.stack((x, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        return x
# Inputs to the model
x = torch.randn(4, 4)
