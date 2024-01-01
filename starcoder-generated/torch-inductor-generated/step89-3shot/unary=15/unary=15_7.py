
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(1, 32, 7, stride=1, padding=2), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.features(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 1, 28, 28)
