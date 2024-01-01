
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 64, 5, stride=1, padding=0), torch.nn.BatchNorm2d(64), torch.nn.ReLU(inplace=False))
    def forward(self, x1):
        v0 = x1
        v1 = self.features(v0)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
