
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.MaxPool2d(3, 2, 1, 1), torch.nn.MaxPool2d(5, 4, 2, 2), torch.nn.MaxPool2d(3, 1, 1, 0))
        self.convolution = torch.nn.Conv2d(3, 3, 3, 1, 1)
    def forward(self, x1):
        v1 = self.features(x1)
        v1 = self.convolution(v1)
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
