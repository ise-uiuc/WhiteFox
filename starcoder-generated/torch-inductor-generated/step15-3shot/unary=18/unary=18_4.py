
class Model(nn.Module):
    def __init__(self, kernel_size=[]):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=1, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=3, stride=1, padding=1)
    def forward(self, x):
        for m in self.modules():
            if type(m)== nn.Conv2d:
                m.weight.data = m.weight.data - 99999.0
                m.bias.data = m.bias.data + 10000
        v1 = self.conv1(x)
        v2 = torch.nn.functional.sigmoid(v1)
        v3 = self.conv2(v2)
        v4 = torch.nn.functional.sigmoid(v3)
        v5 = self.conv3(v4)
        v6 = torch.nn.functional.sigmoid(v5)
        v7 = self.conv4(v6)
        v8 = torch.nn.functional.sigmoid(v7)
        return v8
kernel_size = [1,3,3,3,3,3,3,3,3]
# inputs to the model
x = torch.randn(1,3,64,64)
