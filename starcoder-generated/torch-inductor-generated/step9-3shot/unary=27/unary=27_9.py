
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(15, 100, 33, stride=17, padding=1)
    def forward(self, input):
        v1 = self.conv(input)
        return v1
# Inputs to the model
input = torch.randn(1, 15, 50, 50)
