
class Model(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.conv = torch.nn.Conv2d(17, 10, 7, stride=4, padding=5)
        self.shape = shape
    def forward(self, input):
        v1 = self.conv(input)
        v2 = v1.view(*self.shape)
        return v2
shape = [1, 10, 20]
# Inputs to the model
input = torch.randn(1, 17, 32, 25)
