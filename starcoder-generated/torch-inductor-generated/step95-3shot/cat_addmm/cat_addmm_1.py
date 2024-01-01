
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(2, 2)
        self.conv = nn.Conv2d(2, 2, 2, 1, 0)
    def forward(self, x):
        x = self.layers(x)
        x = self.conv(x)
        x = x.transpose(2, 3).flatten(1)
        return x
# Inputs to the model
x = torch.randn(1, 2, 2, 2)
