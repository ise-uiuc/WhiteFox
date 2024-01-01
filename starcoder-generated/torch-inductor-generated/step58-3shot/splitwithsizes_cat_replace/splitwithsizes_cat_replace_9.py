
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 3, 1, 1, bias=True)
        self.features1 = torch.nn.Conv2d(64, 128, 3, 1, 1, bias=True)
    def forward(self, v1):
        return (torch.nn.Conv2d(448, 1000, 1, 1, 0)(torch.nn.Conv2d(32, 448, 1, 1, 0)(self.features1(self.features(v1)))), torch.split(v1, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
