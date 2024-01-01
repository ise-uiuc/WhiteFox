
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 5, stride=2, padding=2), torch.nn.ReLU(), torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(), torch.nn.Conv2d(64, 128, 3, stride=2, padding=1), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.features(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
