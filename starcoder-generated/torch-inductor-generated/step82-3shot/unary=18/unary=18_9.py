
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, (7, 7), stride=1, padding=3)
        self.conv2 = torch.nn.Conv2d(64, 63, (1, 1), stride=1, padding=0)
        #self.conv3 = torch.nn.Conv2d(3, 256, 1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(63, 5, 1, stride=1, padding=1)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        #v3 = self.conv3(v1)
        #v4 = torch.sigmoid(v3)
        v5 = self.conv4(v2)
        v6 = torch.sigmoid(v5)
        return v6
# Inputs to the model
x1 = torch.randn(1, 3, 224, 224)
