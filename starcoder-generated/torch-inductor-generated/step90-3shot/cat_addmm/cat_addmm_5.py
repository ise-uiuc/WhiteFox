
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 12)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=3)
        x = x.flatten(1)
        x = x.transpose(2, 1)
        return x
# Inputs to the model
x = torch.randn(2, 2, 2)
