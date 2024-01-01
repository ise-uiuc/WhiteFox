
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(12, 12, (8, 4), stride=2, bias=False)
        self.conv2 = torch.nn.ConvTranspose2d(12, 7, (25, 9), stride=1, padding=0, bias=False)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 12, 64, 64)
