
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.ConvTranspose2d(3, 16, 3, stride=1, padding=1)
        self.conv_2 = torch.nn.ConvTranspose2d(16, 16, 5, stride=1, padding=2)
        self.conv_3 = torch.nn.ConvTranspose2d(16, 16, 3, stride=1, padding=1)
        self.conv_4 = torch.nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv_1(x1)
        v2 = self.conv_2(v1)
        v3 = self.conv_3(v2)
        v4 = self.conv_4(v3)
        v5 = torch.relu(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 3, 64, 64)
