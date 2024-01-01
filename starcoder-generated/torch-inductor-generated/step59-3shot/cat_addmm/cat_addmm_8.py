
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 4)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x), dim=1)
        x = torch.cat((x, x), dim=2)
        return x
# Inputs to the model
x = torch.randn(2, 4)
