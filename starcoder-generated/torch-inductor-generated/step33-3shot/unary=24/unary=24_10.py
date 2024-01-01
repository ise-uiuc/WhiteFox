
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(128, 128, 1, stride=1, padding=0)
    def forward(self, x):
        negative_slope = random.uniform(1, 10)
        v1 = self.conv(x)
        v2 = v1 > 0
        v3 = v1 * negative_slope
        v4 = torch.where(v2, v1, v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 128, 38, 38) # TODO: How do I generate a model with a random input size?
