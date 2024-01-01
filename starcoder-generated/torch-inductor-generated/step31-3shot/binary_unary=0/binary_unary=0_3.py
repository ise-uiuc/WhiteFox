
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(24, 48, (1,5), stride=(4,1), padding=(1,2))
        self.conv2 = torch.nn.Conv2d(48, 96, (1,10), stride=(1,5), padding=(0,7))
        self.conv3 = torch.nn.Conv2d(96, 192, (1,20), stride=(1,10), padding=(0,6))
    def forward(self, x):
        v1 = self.conv1(x)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        return v3
# Inputs to the model
x = torch.randn(1, 24, 47, 26)
