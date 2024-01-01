
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv3 = torch.nn.Conv2d(16, 16, 7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(20, 20, 7, stride=1, padding=3)
    def forward(self, x1, x2):
        out1 = self.conv1(x1)
        out1_2 = out1
        out2 = self.conv2(out1_2)
        out2_2 = out2
        out3 = self.conv3(x2)
        out4 = torch.cat([out2_2, out3], 1)
        out5 = self.conv4(out4)
        return out5
# Inputs to the model
x1 = torch.randn(1, 16, 64, 64)
x2 = torch.randn(1, 16, 64, 64)
