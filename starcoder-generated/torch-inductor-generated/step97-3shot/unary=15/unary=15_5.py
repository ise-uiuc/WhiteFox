
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(128, 128, 1, stride=2, padding=0), torch.nn.ReLU())
    def forward(self, x1):
        v1 = self.features(x1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 128, 28, 28)
