
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = torch.split(x, 1, dim=1)
        x = torch.stack(x, dim=1)
        x = torch.sum(x, dim=2).view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
