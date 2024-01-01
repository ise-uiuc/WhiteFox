
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
        self.activation = torch.nn.ReLU6()
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = self.activation(v1)
        return v1
# Inputs to the model
x = torch.randn(1, 3, 256, 256)
