
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Conv2d(3, 32, 33, 11, 17)
    def forward(self, x1, x2):
        v1 = self.features(x1)
        v2 = self.features(x2)
        return (v1, v2)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 32, 32)
