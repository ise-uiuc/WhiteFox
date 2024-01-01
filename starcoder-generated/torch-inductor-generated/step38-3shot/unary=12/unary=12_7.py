
class Model2(torch.nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv2 = torch.nn.Conv2d(3, 4, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = torch.sigmoid(v1)
        v3 = v1 * v2
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
