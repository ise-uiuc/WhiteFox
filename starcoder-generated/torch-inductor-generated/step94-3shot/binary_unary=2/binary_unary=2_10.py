
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        layer0 = torch.nn.Conv2d(3, 64, 5, stride=2, padding=2)
        layer1 = torch.nn.Conv2d(64, 64, 3, stride=2, padding=1)
        layer2 = torch.nn.BatchNorm2d(64)
        self.layers = torch.nn.Sequential(layer2, layer1, layer0)
        self.activation = torch.nn.ReLU()
    def forward(self, x1):
        v1 = self.layers(x1)
        v2 = v1 - 2
        v3 = self.activation(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
