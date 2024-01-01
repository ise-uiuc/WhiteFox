
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Identity(),
            nn.Sigmoid(),
        )
        self.layers1 = nn.Identity()
        
    def forward(self, x):
        x = self.layers(x)
        x = self.layers1(x)
        x = torch.cat((x,x), dim=1).flatten(1, 2)
        return x
# Inputs to the model
x = torch.randn(1, 2)
