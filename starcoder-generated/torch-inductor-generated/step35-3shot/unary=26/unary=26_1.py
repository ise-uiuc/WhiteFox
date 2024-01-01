
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_t = torch.nn.ConvTranspose2d(1, 2, 3, stride=2, padding=1, output_padding=1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
    def forward(self, x3):
        x4 = self.conv_t(x3)
        x5 = self.bn(x4)
        x6 = self.relu(x4)
        return (x5 + x6 + 1.651) > 0.622
# Inputs to the model
x3 = torch.randn(2, 1, 11, 11)
