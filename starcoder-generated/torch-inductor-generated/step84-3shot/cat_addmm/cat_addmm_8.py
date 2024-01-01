
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x), dim=1)
        x = torch.cat((x, x, x), dim=1)
        x = x.transpose(1, 0).flatten(1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
