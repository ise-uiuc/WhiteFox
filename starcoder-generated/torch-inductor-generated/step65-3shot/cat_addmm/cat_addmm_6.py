
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(12, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x), dim=1)
        x = torch.cat((x, x, x, x), dim=0)
        x = torch.mul(x, x)
        return x
