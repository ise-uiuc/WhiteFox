
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 6, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(6, 64, 3, stride=1, padding=1)
        self.conv_3 = torch.nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = torch.max_pool2d(v2, 2, stride=1)
        v4 = self.conv_3(v3)
        v5 = torch.relu(v4)
        v6 = torch.max_pool2d(v5, 3, stride=3)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
