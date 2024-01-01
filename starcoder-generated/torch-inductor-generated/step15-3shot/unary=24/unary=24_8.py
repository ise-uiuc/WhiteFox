
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 2, 3, stride=(2,2), padding=(1,1))
        self.conv2 = torch.nn.Conv2d(2, 4, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(4, 8, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1) > 0
        v3 = self.conv2(v1) * 0.1
        v4 = self.conv3(torch.where(v2, v1, v3))
        return v4
# Inputs to the model
x1 = torch.randn(1, 1, 512, 512)
