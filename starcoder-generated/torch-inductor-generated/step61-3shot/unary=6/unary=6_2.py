
class Model(torch.nn.AvgPool2d):
    def __init__(self):
        super().__init__(1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self(x1)
        v2 = v1 + 3
        v3 = torch.clamp(v2, 0, 6)
        v4 = v1 * v3
        v5 = v4 / 6
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 256, 256)
