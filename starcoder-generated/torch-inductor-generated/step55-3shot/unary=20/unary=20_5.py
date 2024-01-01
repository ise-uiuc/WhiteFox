
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1)
        self.conv = torch.nn.Conv2d(3, 3, kernel_size=3, stride=1)
    def forward(self, x1):
        v1 = self.conv_t(x1)
        v2 = self.conv(v1)
        v3 = torch.nn.functional.adaptive_avg_pool2d(v2, output_size=[3, 3])
        v4 = torch.flatten(v3, 1)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 20, 50)
