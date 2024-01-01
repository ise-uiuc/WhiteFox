
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, (3, 3), 1, (2, 1))
        self.conv2 = torch.nn.ConvTranspose2d(3, 3, (1, 3), groups=2, padding=(2, 1))
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = torch.sigmoid(self.conv2(v2))
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 9, 28)
