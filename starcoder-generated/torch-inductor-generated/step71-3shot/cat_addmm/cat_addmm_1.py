
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 5)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x, x, x, x, x, x), dim=1)
        x = torch.sum(x, dim=0)
        return x
# Inputs to the model
x = torch.randn(3, 5)
