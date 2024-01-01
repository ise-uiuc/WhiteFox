
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 5)
    def forward(self, x):
        x = torch.cat((x, x, x), dim=0)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(2, 3)
