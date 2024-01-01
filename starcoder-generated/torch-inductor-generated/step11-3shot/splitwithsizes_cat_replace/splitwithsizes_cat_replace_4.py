
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 1), torch.nn.Conv2d(32, 3, 3, 1, 1))
        self.classifier = torch.nn.Sequential(torch.nn.Linear(3 * 64 * 64, 10))
    def forward(self, x0):
        v0 = self.features(x0)
        v1 = self.classifier(v0.view(1, 3 * 64 * 64))
        return (v1, torch.split(v0, [1, 1, 1], dim=1))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
