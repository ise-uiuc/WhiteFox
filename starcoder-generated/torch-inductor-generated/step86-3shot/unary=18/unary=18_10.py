
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=240, kernel_size=9, padding=1)
        self.conv2 = torch.nn.ConvTranspose2d(in_channels=240, out_channels=3, kernel_size=9, stride=6, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = torch.sigmoid(v1)
        v3 = self.conv2(v2)   
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 300, 300)
