
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 3, stride=3, padding=4)
    def forward(self, x1):
        v1 = torch.nn.functional.interpolate(x1, scale_factor=2)
        v2 = torch.nn.functional.leaky_relu(self.conv1(v1))
        return v2
# Inputs to the model
x1 = torch.randn(1, 16, 226, 226)
