
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(2, 4)
        self.layers2 = nn.Linear(3, 8)
    def forward(self, x):
        x = self.layers1(x)
        x1 = torch.cat([x, x], dim=0)
        x = self.layers2(x1)
        x = torch.stack((x, x), dim=1)
        x = torch.flatten(x, start_dim=1)
        x = x[0]
        return x
# Inputs to the model
x = torch.randn(2, 2)
