
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 9)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = x.flatten(start_dim=1)
        x = x.view(x.shape[0], x.shape[1], 2, 2)
        return x
# Inputs to the model
x = torch.randn(1, 2)
