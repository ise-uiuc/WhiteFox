
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=0)
        x = x.flatten(1)
        x = x.transpose(1, 0)
        x = x[0]
        return x
# Inputs to the model
x = torch.randn(2, 3)
