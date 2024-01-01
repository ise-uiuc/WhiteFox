
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 4, 1, 1, 0)
        self.conv_2 = torch.nn.ConvTranspose2d(4, 16, 2, 1, 3)
        self.conv_3 = torch.nn.ConvTranspose2d(32, 16, 3, 1, 2, output_padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = torch.avg_pool2d(v1, 2, stride=1)
        v3 = self.conv_2(v2)
        v4 = self.conv_3(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
