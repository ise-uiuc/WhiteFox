
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 9, 5, stride=2, padding=2)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 - 0.5
        v3 = F.leaky_relu(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 4, 512, 512)
