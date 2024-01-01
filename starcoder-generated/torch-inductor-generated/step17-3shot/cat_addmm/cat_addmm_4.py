
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        x = torch.cat((x, x, x, x), dim=0)
        return x
# Inputs to the model
x = torch.randn(4, 4)
