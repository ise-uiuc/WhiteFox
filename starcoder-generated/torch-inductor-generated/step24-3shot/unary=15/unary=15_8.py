
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features2 = torch.nn.ReLU()
        self.features1 = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.features3 = torch.nn.AvgPool2d((3, 3), stride=2, padding=1)
    def forward(self, x1):
        v1 = self.features1(x1)
        v2 = self.features2(v1)
        v3 = self.features3(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
