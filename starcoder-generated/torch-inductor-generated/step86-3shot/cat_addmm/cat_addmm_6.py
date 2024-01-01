
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Linear(4, 100)
        self.layers2 = nn.Linear(4, 100)
    def forward(self, x):
        x = self.layers1(x)
        x = torch.cat((x, x, x, x), dim=1)
        x = self.layers2(x)
        return x
# Inputs to the model
x = torch.randn(2, 4)
