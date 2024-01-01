
class Model(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(3,2,2,stride=1,padding=3))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.ReLU(inplace=True))
        self.min = min
        self.max = max
        self.layers=torch.nn.Sequential(*layers)
    def forward(self, x1):
        v1 = self.layers(x1)
        v2 = torch.clamp_min(v1, self.min)
        v3 = torch.clamp_max(v2, self.max)
        return v3
min = 5
max = 3
# Inputs to the model
x1 = torch.randn(1, 3, 7, 7)
