
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 7, 1, stride=1, padding=1)
    def forward(self, input):
        return self.conv(input) - 0.37
# Inputs to the model
input = torch.randn(1, 3, 22, 22)
