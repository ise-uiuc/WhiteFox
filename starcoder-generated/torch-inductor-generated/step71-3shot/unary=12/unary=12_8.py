
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 16, 7, stride=1, padding=3)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x1):
        v1 = self.conv(x1)
    def forward(self, x1):
        v2 = self.sigmoid(v1)
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
