
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(32, 64, 4, stride=1, padding=2, dilation=2)
    def forward(self, input):
        v1 = self.conv1(input)
        v2 = torch.tanh(v1)
        return v1
# Inputs to the model
x = torch.randn(2, 32, 32, 32)
