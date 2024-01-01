
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 112)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack([x] * 56, dim=1)
        x = torch.cat((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(1, 6)
