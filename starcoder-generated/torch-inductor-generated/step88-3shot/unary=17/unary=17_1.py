
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 1, 3, stride=2,padding=1, output_padding=1)
        self.conv1 = torch.nn.Conv2d(1, 1, 3, stride=2,padding=0, output_padding=0)
        self.conv_1 = torch.nn.ConvTranspose2d(1, 1, 1, stride=2,padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = F.relu(v1)
        v3 = self.conv1(v2)
        v4 = torch.max(v3)
        v5 = self.conv_1(v4)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 1, 3, 3)
