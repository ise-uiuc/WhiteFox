
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2 = torch.nn.Conv2d(1, 64, 3, stride=1, padding='same')
    def forward(self, x1):
        v1 = self.conv2(x1)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 1, 64, 64)
