
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(3, 1, (5, 5), stride=1, padding=1)
    def forward(self, x1):
        v2 = self.conv2(x1)
        v1 = torch.sigmoid(v2)
        v3 = v2 * v1
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 32, 32)
