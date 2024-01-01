
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(3, 3)
    def forward(self, x):
        x = self.layers(x)
        x1 = torch.chunk(x, chunks=2, dim=1)
        return x1
# Inputs to the model
x = torch.randn(2, 3)
