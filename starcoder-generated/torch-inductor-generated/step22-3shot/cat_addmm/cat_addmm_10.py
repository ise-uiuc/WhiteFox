
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 512)
    def forward(self, x):
        x = self.layers(x).reshape(-1, 32, 32, 1)
        x = torch.stack((x, x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 1)
