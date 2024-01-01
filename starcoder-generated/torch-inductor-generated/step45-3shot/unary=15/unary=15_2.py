
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv11 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv12 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv13 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv14 = torch.nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.conv15 = torch.nn.Conv2d(512, 101, 1, stride=1, padding=0)
    def forward(self, x1):
        v1 = self.conv1(x1)
        v2 = self.conv2(v1)
        v3 = self.conv3(v2)
        v4 = self.conv4(v3)
        v5 = self.conv5(v4)
        v6 = self.conv6(v5)
        v7 = self.conv7(v6)
        v8 = self.conv8(v7)
        v9 = self.conv9(v8)
        v10 = self.conv10(v9)
        v11 = self.conv11(v10)
        v12 = self.conv12(v11)
        v13 = self.conv13(v12)
        v14 = self.conv14(v13)
        v15 = self.conv15(v14)
        v16 = torch.softmax(v15, dim=1)
        return v16
# Inputs to the model
x1 = torch.randn(1, 512, 10, 10)
