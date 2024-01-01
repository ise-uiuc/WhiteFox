
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 5)
        self.linear1 = nn.Linear(3, 6)
        self.linear2 = nn.Linear(3, 7)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat([x, x, x], dim=1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
# Inputs to the model
x = torch.randn(3, 3)
