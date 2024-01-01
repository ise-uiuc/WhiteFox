
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 4, 1, stride=2, padding=2)
    def forward(self, input):
        v1 = torch.tanh(self.conv(input))
        v2 = torch.tanh(self.conv(input))
        v3 = v1 * v2
        return v3
# Inputs to the model
x = torch.randn(1, 3, 64, 64)
