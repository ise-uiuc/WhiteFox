
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
    def forward(self, x):
        x = self.layers(x)
        return torch.cat((x, x), dim=1)
# Inputs to the model
x = torch.randn(2, 2)
