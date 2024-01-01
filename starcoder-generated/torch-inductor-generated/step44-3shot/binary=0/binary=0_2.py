
class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(in_channels=2, out_channels=4, kernel_size=1, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=1, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(in_channels=10, out_channels=4, kernel_size=1, stride=1, padding=1)
    def forward(self, x1, padding5=True, x2=True, x3=True, padding4=True, padding2=True, padding3=None):
        v1 = self.conv1(x1)
        if x2 == True:
            x2 = torch.randn(x1.shape)
        v2 = self.conv2(x2)
        if v1[0][0][0][0] == 1.0:
            v1 = torch.randn(v1.shape)
        if padding2 == True:
            padding2 = torch.randn(v2.shape)
        v3 = v1 + v2
        v4 = self.conv3(v3)
        if padding3 == None:
            padding3 = torch.randn(v4.shape)
        v5 = self.conv4(v4)
        if x3 == True:
            x3 = torch.randn(x1.shape)
        v6 = self.conv5(torch.cat((v5, x3)))
        if padding5 == True:
            padding5 = torch.randn(v6.shape)
        v7 = torch.cat((v6, padding5))
        v8 = self.conv5(torch.cat((v7, v8)))
        return v8
# Inputs to the model
x1 = torch.randn(4, 4, 32, 32)
