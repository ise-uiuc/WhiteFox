
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Linear(5, 4)
    def forward(self, x, w):
        x = self.layers(x)
        y = torch.stack((x, x), dim=1).flatten(1)
        if torch.numel(y) == 8:
            y = torch.neg(y)
        z = torch.stack((y, x, y, w), dim=0)
        z = z.flatten(0)
        return z
# Inputs to the model
x = torch.randn(4, 5)
w = torch.randn(2, 2)
