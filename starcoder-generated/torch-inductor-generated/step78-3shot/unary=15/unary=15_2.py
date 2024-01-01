
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d((5, 5)), torch.nn.Conv2d(3, 3, (1, 1)), torch.nn.ReLU())
    def forward(self, x1):
        v0 = x1
        v1 = self.features(v0)
        v2 = torch.relu(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
