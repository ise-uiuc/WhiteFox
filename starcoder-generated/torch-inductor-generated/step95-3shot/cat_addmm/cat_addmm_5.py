
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 1)
    def forward(self, x):
        x = torch.cat((x, x), dim=1).flatten(start_dim=1)
        x = self.layers(x)
        return x
# Inputs to the model
x = torch.randn(1, 2)
