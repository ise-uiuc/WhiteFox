
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(4, 2)
    def forward(self, x):
        x = self.layers(x)
        x = x * 0.3
        x = torch.cat((x, x), dim=1)
        x = x - 0.3
        return x
# Inputs to the model
x = torch.randn(2, 4)
