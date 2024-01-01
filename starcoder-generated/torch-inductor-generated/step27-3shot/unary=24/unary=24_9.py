
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(128, 64, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 1, 7, stride=1, padding=0)
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv1(v1)
        v3 = self.conv2(v2)
        v4 = v3 > 0.5
        v5 = torch.where(v4, torch.tensor(1.), torch.tensor(0.))
        return v5
# Inputs to the model
x1 = torch.rand(1, 128, 224, 224)
