
class Model(torch.nn.Sequential):
    def __init__(self):
        super().__init__()
        self.add_module('conv', torch.nn.ConvTranspose2d(3, 3, 3, stride=2, padding=1))
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = torch.relu(v1)
        v3 = torch.sigmoid(v2)
        v4 = torch.tanh(v3)
        return v4
# Inputs to the model
x1 = torch.randn(1, 3, 16, 16)
