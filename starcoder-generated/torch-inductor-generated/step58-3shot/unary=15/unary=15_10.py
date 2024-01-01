
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 32, 7, stride=2, padding=3)
        self.conv2 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1)
    def forward(self, x1):
        out = self.conv1(x1)
        out = self.conv2(out)
        out = torch.reshape(out, (out.shape[0], -1))
        return out
# Inputs to the model
x1 = torch.randn(2, 2, 128, 128)
