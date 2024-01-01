
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.sigmoid = torch.sigmoid
    def forward(self, x1):
        v1 = self.conv2d(x1)
        v2 = self.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
