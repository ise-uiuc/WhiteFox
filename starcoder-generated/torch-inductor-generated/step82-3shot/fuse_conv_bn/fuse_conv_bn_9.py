
class Model(torch.nn.Module):
    def __init__(self, x):
        super().__init__()
        self.features = torch.nn.Sequential(torch.nn.Conv2d(4, 4, 3, bias=True), torch.nn.ReLU6(), torch.nn.BatchNorm2d(32), torch.nn.ReLU6(), torch.nn.Conv2d(32, 32, 3, bias=True), torch.nn.ReLU6(), torch.nn.BatchNorm2d(4))
    def forward(self, x):
        s1 = self.features(x)
        return s1

# Inputs to the model
x = torch.randn(1, 4, 5, 5)
