
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.sum(dim=1, keepdim=True)
        y = torch.stack((x, x, x, x), dim=1)
        x = y.flatten(start_dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
