
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 16)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=1)
        x = torch.stack((x, x, x, x), dim=2)
        x = torch.flatten(x, start_dim=2)
        x = x.mean(dim=2).view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
