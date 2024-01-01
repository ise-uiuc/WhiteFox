
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 3, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv(x1)
        v1 = torch.reshape(v1, (1, 9))
        v1 = v1.flatten()
        return v1
# Inputs to the model
x1 = torch.randn(1, 3, 1024, 1024)
