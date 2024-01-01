
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(torch.nn.Conv2d(2, 2, 2, padding=1), nn.Linear(4, 6))
    def forward(self, x):
        x = self.layers(x)
        #x = x.flatten(start_dim=1)
        #x = torch.stack([x, x], dim=1)
        #x = x.flatten(start_dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2, 2)
