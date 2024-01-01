
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose = torch.nn.ConvTranspose2d(15, 12, 1, padding=0, stride=1)
        self.conv0 = torch.nn.Conv2d(5, 56, 8, padding=0, stride=8)
        self.conv1 = torch.nn.Conv2d(13, 1, 8, padding=0, stride=8)
    def forward(self, x1):
        v1 = self.conv_transpose(x1)
        v2 = self.conv0(v1)
        v3 = torch.relu(v2)
        v4 = self.conv1(v3)
        v5 = torch.sigmoid(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 15, 1, 1)
