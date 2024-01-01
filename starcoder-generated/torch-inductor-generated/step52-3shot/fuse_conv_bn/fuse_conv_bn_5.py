
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(512, 512, (3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.bn_1 = torch.nn.BatchNorm2d(512, track_running_stats=True)
        self.conv_2 = torch.nn.Conv2d(512, 256, (1, 1), stride=(1, 1), bias=False)
        self.bn_2 = torch.nn.BatchNorm2d(256, track_running_stats=True)
    def forward(self, x3):
        x = self.conv_1(x3)
        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.bn_2(x)
        return x3 + x + x
# Inputs to the model
x3 = torch.randn(1, 512, 8, 8)
