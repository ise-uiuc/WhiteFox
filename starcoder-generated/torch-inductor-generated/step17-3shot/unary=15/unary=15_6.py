
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = torch.nn.Sequential(torch.nn.Conv2d(3, 32, 1, stride=1, padding=0), torch.nn.MaxPool2d(3, stride=2, padding=0), torch.nn.ReLU())
        self.m2 = torch.nn.Sequential(torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32), torch.nn.Conv2d(32, 32, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(32), torch.nn.MaxPool2d(3, stride=2, padding=1))
        self.m3 = torch.nn.Sequential(torch.nn.Conv2d(32, 64, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(64), torch.nn.Conv2d(64, 64, 3, stride=1, padding=1), torch.nn.ReLU(), torch.nn.BatchNorm2d(64), torch.nn.MaxPool2d(3, stride=2, padding=1))
    def forward(self, x1):
        v1 = self.m1(x1)
        v2 = self.m2(v1)
        v3 = self.m3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
