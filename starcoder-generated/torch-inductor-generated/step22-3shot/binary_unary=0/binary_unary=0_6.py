
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 1, 5, stride=1, padding=2)
    def forward(self, x):
        v1 = self.conv(x)
        v3 = torch.sigmoid(v1)
        return v3
# Inputs to the model
x = torch.randn(1, 4, 128, 128)
