
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 3)
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(0, 1).flatten(0, 1)
        x = x.contiguous().flatten(0, 1)
        x = torch.stack([x])
        return x
x = torch.randn(2, 2)
