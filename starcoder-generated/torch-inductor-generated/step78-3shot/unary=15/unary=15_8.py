
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 3, (1, 1)), torch.nn.ReLU(), torch.nn.Conv2d(3, 3, (1, 1)), torch.nn.ReLU(), torch.nn.Conv2d(3, 3, (1, 1)))
    def forward(self, x1):
        v1 = self.features(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
