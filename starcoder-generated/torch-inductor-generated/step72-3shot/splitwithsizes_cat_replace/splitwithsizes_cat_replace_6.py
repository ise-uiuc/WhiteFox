
class InputLayerSqueeze(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = torch.nn.AdaptiveAvgPool2d((7, 7))
    def forward(self, v1):
        return self.op(v1).squeeze()
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = InputLayerSqueeze()
    def forward(self, v1, v2):
        return (self.features(v1), self.features(v2))
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
x2 = torch.randn(1, 3, 64, 64)
