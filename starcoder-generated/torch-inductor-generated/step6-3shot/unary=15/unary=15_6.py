
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features2 = torch.nn.ReLU()
        self.features1 = torch.nn.Conv2d(3, 8, 3, stride=2, padding=1)
    def forward(self, x1):
        v1 = self.features1(x1)
        v2 = self.features2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
