
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(6, 13, 5, stride=2, padding=2)
        self.conv2 = torch.nn.ConvTranspose2d(13, 7, 7, stride=2, padding=2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        return v2
# Inputs to the model
x1 = torch.randn(1, 6, 16, 16)
