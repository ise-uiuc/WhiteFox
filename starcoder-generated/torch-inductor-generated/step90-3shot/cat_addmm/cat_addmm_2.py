
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
        self.cat = torch.cat

    def forward(self, x):
        x1 = self.layers(x)
        x2 = x1.view((1, 2, 2))
        x1 = torch.cat((x1, x1), dim=0)
        x = torch.cat([x1, x2], dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 4)
