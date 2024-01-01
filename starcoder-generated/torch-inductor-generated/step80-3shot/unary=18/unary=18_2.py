
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, 2, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v1 = torch.transpose(v1, 1, 3)
        v2 = torch.sigmoid(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 3, 256, 128)
