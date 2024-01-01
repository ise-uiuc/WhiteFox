
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 3, 1, 2))
        self.split = torch.nn.Sequential(torch.nn.AvgPool2d(2, 1, 1, 0), torch.nn.MaxPool2d(2, 1, 2, 2))
    def forward(self, x1):
        v1 = self.features(x1)
        return [v1, self.split(x1)]
# Inputs to the model
x1 = torch.randn(1, 3, 56, 56)
