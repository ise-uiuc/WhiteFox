
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 15, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(15, 100, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(100, 550, (5, 5), stride=1, padding=3, dilation=2)
        self.conv4 = torch.nn.Conv2d(550, 20, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(20, 25, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = v1 * 0.5
        v3 = v1 * 0.7071067811865476
        v4 = torch.erf(v3)
        v5 = v4 + 1
        v6 = v2 * v5
        v7 = self.conv2(v6)
        v8 = torch.transpose(self.conv3(v7),1,2)
        v9 = torch.max(v8, dim=1)[0]
        v10 = torch.nn.functional.interpolate(v9, size=[225,225])
        v11 = self.conv4(v10)
        v12 = torch.max(self.conv5(v11), dim=3)[0]
        v765 = torch.flatten(torch.transpose(v12, 1, 2))
        return v765
# Inputs to the model
x1 = torch.randn(10, 1, 10, 10)
