
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 250)
        self.cat = torch.cat
    def forward(self, x):
        x = self.layers(x)
        x = self.cat([x]*150, dim=1)
        x = x.view(x.shape[0], -1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
