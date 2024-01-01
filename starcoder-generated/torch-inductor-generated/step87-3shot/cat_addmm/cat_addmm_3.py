
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(1, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.stack((x, x, x, x), dim=0)
        x = torch.zeros_like(x)
        x = x[3]
        return x
# Inputs to the model
x = torch.randn(2, 1)
