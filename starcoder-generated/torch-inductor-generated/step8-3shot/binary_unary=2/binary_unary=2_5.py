
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(1, 8, 4, stride=2, padding=1, bias=True)
        self.conv2 = torch.nn.ConvTranspose2d(8, 4, 5, stride=2, padding=1, bias=True)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = v2 * 0.5
        return v3
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
