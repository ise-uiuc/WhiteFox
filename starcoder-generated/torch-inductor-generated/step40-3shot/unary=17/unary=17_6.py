
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 64, kernel_size=2)
        self.conv_2 = torch.nn.ConvTranspose2d(64, 128, kernel_size=2, stride=2)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = F.relu(v1)
        v2 = v2.max_pool2d(3, stride=3)
        v3 = self.conv_2(v2)
        return v3
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
