
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 4)
        self.tanh = nn.Tanh()
    def forward(self, x):
        x = self.layers(x)
        x = self.tanh(x)
        x = torch.stack((x, x), dim=1)
        return x
# Inputs to the model
x = torch.randn(2, 2)
