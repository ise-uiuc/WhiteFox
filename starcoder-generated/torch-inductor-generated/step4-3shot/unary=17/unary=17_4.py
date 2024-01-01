
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(10, 8, 1, stride=1, padding=0)
        self.conv_transpose = torch.nn.ConvTranspose2d(3, 8, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(8, 200, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = self.conv_transpose(v1)
        v3 = F.relu(v2)
        v4 = self.conv2(v3)
        v5 = F.softmax(v4)
        return v5
# Inputs to the model
x1 = torch.randn(1, 10, 64, 64)
