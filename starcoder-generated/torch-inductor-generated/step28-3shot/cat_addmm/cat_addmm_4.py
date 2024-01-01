
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 20)
        self.linear3 = nn.Linear(30, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x, x, x], dim=1)
        x = torch.stack([x, x, x], dim=1)
        x = torch.stack([x, x, x], dim=1)
        x = torch.stack([x, x, x], dim=1)
        x = x.flatten(start_dim=1)
        x = self.linear3(x)
        return x
# Inputs to the model
x = torch.randn(2, 2)
