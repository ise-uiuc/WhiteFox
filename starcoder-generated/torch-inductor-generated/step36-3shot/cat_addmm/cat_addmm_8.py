
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(257, 1)
    def forward(self, x):
        x = self.layers(x)
        x = torch.split(x, 6, dim=1)
        x = torch.cat(x, 1)
        return x
# Inputs to the model
x = torch.randn(1, 257)
