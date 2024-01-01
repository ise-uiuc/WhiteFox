
class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 2, 1, stride=2, padding=2)

    def forward(self, input):
        t1 = self.conv(input)
        t2 = t1 - 1.0
        return t2
# Inputs to the model
input = torch.randn(1, 3, 28, 28)
