
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(6, 6)
    def forward(self, x):
        x = torch.stack((x, x), dim=1)
        x = self.layers(x).transpose(-1, -2)
        return x
# Inputs to the model
x = torch.randn(2, 6)
