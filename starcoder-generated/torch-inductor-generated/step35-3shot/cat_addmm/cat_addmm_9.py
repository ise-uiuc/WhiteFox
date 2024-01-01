
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        if not None:
            x = torch.flatten(x, 1)
            x = x.flatten()
        x = torch.stack((x, x), dim=0)
        x = x.flatten(1)
        x = x.unsqueeze(0)
        x = x.flatten()
        return x
# Inputs to the model
x = torch.randn(1, 2)
