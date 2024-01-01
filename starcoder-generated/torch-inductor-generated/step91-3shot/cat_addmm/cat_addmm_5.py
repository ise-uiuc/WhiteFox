
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(7, 6)
        self.stack = torch.stack
    def forward(self, x):
        x = self.layers(x)
        x = self.stack((x, x), dim=1)
        x = x.flatten(start_dim=1)
        if bool(random.getrandbits(1)):
            x = x.T
        return x
# Inputs to the model
x = torch.randn(7, 7)
