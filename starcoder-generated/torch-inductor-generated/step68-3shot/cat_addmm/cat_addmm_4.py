
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 5)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten()
        x = torch.stack((x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=1)
        x = torch.stack((x, x), dim=1).flatten()
        x = torch.dot(x, x)
        return x
# Inputs to the model
x = torch.arange(8).view(2, 4)
