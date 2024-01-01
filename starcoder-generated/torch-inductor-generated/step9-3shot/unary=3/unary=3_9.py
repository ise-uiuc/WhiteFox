
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(10, 2, 5, padding=(2, 2))
        self.conv2 = torch.nn.Conv2d(2, 2, 1, stride=1, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = out * 0.5
        out = out * 0.7071067811865476
        out = torch.erf(out)
        out = out + 1
        out = out * self.conv2(out)
        return out
# Inputs to the model
# Please note that input must be padded to be 4 * 10 + 4 = 84 x 44 pixels, and the output of the model is a 1x1 grid
x1 = torch.randn(1, 10, 40, 40)
